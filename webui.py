import os
import sys
import json
import time
import re
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import streamlit as st
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from text_splitter import TextSplitter
from debug_utils import DebugLogger

class EnhancedLLMClient:
    def __init__(self, api_key: str, api_url: str = "https://api.deepseek.com/v1/chat/completions", debug_logger=None):
        """初始化增强版LLM客户端，增加多线程支持"""
        import requests
        self.requests = requests
        
        self.api_key = api_key
        if not self.api_key:
            raise ValueError("API密钥未设置")
            
        self.api_url = api_url
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # 加载prompt模板
        self.prompt_template = None
        self.chapter_prompt_template = None
        
        # 初始化debug日志记录器
        self.debug_logger = debug_logger
        
        # 请求队列和结果队列，用于多线程处理
        self.request_queue = queue.Queue()
        self.result_queue = queue.Queue()
        
        # 最大并发数
        self.max_workers = 5
        
        # 线程池
        self.executor = None
        
        # 标志变量，用于控制线程池
        self.running = False
    
    def set_max_workers(self, max_workers: int):
        """设置最大并发数"""
        self.max_workers = max_workers
    
    def load_prompt_template(self, prompt_file_path: str):
        """加载prompt模板"""
        try:
            with open(prompt_file_path, "r", encoding="utf-8") as f:
                self.prompt_template = f.read()
        except Exception as e:
            print(f"加载prompt模板失败: {e}")
            raise
    
    def load_chapter_prompt_template(self, chapter_prompt_file_path: str):
        """加载章节prompt模板"""
        try:
            with open(chapter_prompt_file_path, "r", encoding="utf-8") as f:
                self.chapter_prompt_template = f.read()
        except Exception as e:
            print(f"加载章节prompt模板失败: {e}")
            raise
    
    def _call_api(self, data: dict, request_type: str, text: str) -> str:
        """实际调用API的函数，用于线程池"""
        try:
            # 如果有debug_logger，写入请求内容
            if self.debug_logger:
                self.debug_logger.write_debuglog(request_type, text, "待发送")
                
            response = self.requests.post(self.api_url, headers=self.headers, json=data, timeout=60)
            response.raise_for_status()
            
            # 解析响应
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                llm_response = result["choices"][0]["message"]["content"]
                # 如果有debug_logger，写入响应内容
                if self.debug_logger:
                    self.debug_logger.write_debuglog(request_type, text, llm_response)
                return llm_response
            else:
                error_msg = "API response format error"
                if self.debug_logger:
                    self.debug_logger.write_debuglog(request_type, text, error_msg)
                return ""
                
        except Exception as e:
            print(f"API调用错误: {e}")
            # 重试一次
            time.sleep(2)
            try:
                response = self.requests.post(self.api_url, headers=self.headers, json=data, timeout=60)
                response.raise_for_status()
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    llm_response = result["choices"][0]["message"]["content"]
                    # 如果有debug_logger，写入重试响应内容
                    if self.debug_logger:
                        self.debug_logger.write_debuglog(request_type, text, llm_response)
                    return llm_response
                else:
                    error_msg = "API response format error after retry"
                    if self.debug_logger:
                        self.debug_logger.write_debuglog(request_type, text, error_msg)
                    return ""
            except Exception as e2:
                print(f"API重试失败: {e2}")
                if self.debug_logger:
                    self.debug_logger.write_debuglog(request_type, text, str(e2))
                return ""
    
    def call_llm(self, text: str) -> str:
        """调用API处理文本"""
        if not self.prompt_template:
            raise ValueError("prompt模板未加载")
        
        messages = [
            {"role": "system", "content": self.prompt_template},
            {"role": "user", "content": text}
        ]
        
        data = {
            "model": "deepseek-chat",
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 8192
        }
        
        return self._call_api(data, "JSON生成请求", text)
    
    def call_llm_for_chapter_roles(self, text: str) -> str:
        """调用API获取章节角色信息"""
        if not self.chapter_prompt_template:
            raise ValueError("章节prompt模板未加载")
        
        messages = [
            {"role": "system", "content": self.chapter_prompt_template},
            {"role": "user", "content": text}
        ]
        
        data = {
            "model": "deepseek-chat",
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1024
        }
        
        return self._call_api(data, "章节角色信息请求", text)
    
    def extract_json_from_response(self, response: str) -> list:
        """从LLM响应中提取JSON内容"""
        try:
            # 尝试直接解析响应
            return json.loads(response)
        except json.JSONDecodeError:
            # 如果失败，尝试提取JSON代码块
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
            if json_match:
                try:
                    return json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    print("无法解析提取的JSON")
                    return []
            else:
                print("响应中未找到JSON")
                return []
    
    def extract_roles_from_response(self, response: str) -> list:
        """从LLM响应中提取角色列表"""
        try:
            # 尝试直接解析响应
            roles = json.loads(response)
            if isinstance(roles, list):
                return roles
            else:
                # 如果是字典，尝试提取角色列表
                if isinstance(roles, dict) and 'roles' in roles:
                    return roles['roles']
        except json.JSONDecodeError:
            # 尝试从文本中提取角色
            role_pattern = r'(?:角色列表|人物列表)[:：]?\s*[\[\(]?([^\]\)]*)'
            match = re.search(role_pattern, response)
            if match:
                roles_text = match.group(1)
                # 分割角色列表
                roles = [role.strip() for role in re.split(r'[，,、\s]+', roles_text) if role.strip()]
                return roles
        return []
    
    def process_segment_with_threadpool(self, segment: str, segment_id: int, chapter_roles: list = None):
        """使用线程池处理单个片段"""
        segment_with_roles = segment
        if chapter_roles:
            roles_text = ",".join(chapter_roles)
            segment_with_roles = f"从以下角色中选取这段文本的角色，请注意这段文本只是章节片段，不是所有角色一定都要出现。角色列表：{roles_text}\n\n{segment}"
        
        llm_response = self.call_llm(segment_with_roles)
        if not llm_response:
            self.result_queue.put((segment_id, None))
            return
        
        segment_json = self.extract_json_from_response(llm_response)
        self.result_queue.put((segment_id, segment_json if segment_json else None))

class WebUITxtToJsonConverter:
    def __init__(self, base_dir: str):
        # 初始化基础目录和相关路径
        self.base_dir = base_dir
        
        # 使用一个临时的默认名称，实际处理文件时会更新
        text_filename = "temp_text"
        
        # 创建\workshop\文本同名文件夹目录结构
        workshop_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workshop")
        text_specific_dir = os.path.join(workshop_dir, text_filename)
        
        # 创建目录
        self.text_segments_dir = os.path.join(text_specific_dir, "text_segments")
        self.json_results_dir = os.path.join(text_specific_dir, "scripts")
        os.makedirs(self.text_segments_dir, exist_ok=True)
        os.makedirs(self.json_results_dir, exist_ok=True)
        
        # 初始化拆分器和调试日志器
        self.splitter = TextSplitter()
        
        # 先初始化debug_logger
        self.debug_logger = DebugLogger(os.path.join(text_specific_dir, "debug.log"))
        
        # LLM客户端将在界面中配置
        self.llm_client = None
        
        # 配置参数
        self.max_workers = 5
        self.max_segment_length = 350
        self.output_dir = text_specific_dir
        
    def update_directory_by_filename(self, input_file: str):
        """根据输入文件名更新目录结构"""
        # 获取输入文件的文件名（不含扩展名）作为文本同名文件夹名称
        text_filename = os.path.splitext(os.path.basename(input_file))[0]
        
        # 更新\workshop\文本同名文件夹目录结构
        workshop_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workshop")
        text_specific_dir = os.path.join(workshop_dir, text_filename)
        
        # 更新目录
        self.text_segments_dir = os.path.join(text_specific_dir, "text_segments")
        self.json_results_dir = os.path.join(text_specific_dir, "scripts")
        os.makedirs(self.text_segments_dir, exist_ok=True)
        os.makedirs(self.json_results_dir, exist_ok=True)
        
        # 创建project.json文件
        project_json_path = os.path.join(text_specific_dir, "project.json")
        project_data = {
            "projectName": text_filename
        }
        with open(project_json_path, "w", encoding="utf-8") as f:
            json.dump(project_data, f, ensure_ascii=False, indent=2)
        
        # 更新调试日志器路径
        self.debug_logger = DebugLogger(os.path.join(text_specific_dir, "debug.log"))
        
        # 更新输出目录
        self.output_dir = text_specific_dir

    def setup_llm_client(self, api_key: str, api_url: str = "https://api.deepseek.com/v1/chat/completions"):
        """设置LLM客户端"""
        self.llm_client = EnhancedLLMClient(
            api_key=api_key,
            api_url=api_url,
            debug_logger=self.debug_logger
        )
        self.llm_client.set_max_workers(self.max_workers)
        
        # 加载prompt模板
        prompt_file = os.path.join(self.base_dir, "json生成prompt")
        chapter_prompt_file = os.path.join(self.base_dir, "章节prompt")
        
        if os.path.exists(prompt_file):
            self.llm_client.load_prompt_template(prompt_file)
        else:
            st.warning(f"提示模板文件不存在: {prompt_file}")
        
        if os.path.exists(chapter_prompt_file):
            self.llm_client.load_chapter_prompt_template(chapter_prompt_file)
        else:
            st.warning(f"章节提示模板文件不存在: {chapter_prompt_file}")
        
    def process_text(self, text: str, progress_bar: Optional[st.progress] = None, status_text: Optional[st.empty] = None):
        """处理完整文本，返回按章节组织的JSON结果"""
        if not self.llm_client:
            raise ValueError("LLM客户端未初始化，请先设置API密钥")
        
        # 第一步：按章节分割
        chapters = self.splitter.split_by_chapter(text)
        result = {}
        total_chapters = len(chapters)
        processed_chapters = 0
        
        for chapter_title, chapter_content in chapters:
            if status_text:
                status_text.text(f"处理章节: {chapter_title}")
            
            # 保存原始章节文本
            safe_title = re.sub(r'[\\/:*?"<>|]', '_', chapter_title)
            chapter_text_file = os.path.join(self.text_segments_dir, f"{safe_title}.txt")
            with open(chapter_text_file, "w", encoding="utf-8") as f:
                f.write(chapter_content)
            
            # 获取章节角色信息
            if status_text:
                status_text.text(f"处理章节: {chapter_title} - 获取角色信息...")
                
            chapter_roles_response = self.llm_client.call_llm_for_chapter_roles(chapter_content)
            chapter_roles = []
            if chapter_roles_response:
                chapter_roles = self.llm_client.extract_roles_from_response(chapter_roles_response)
                if chapter_roles:
                    if status_text:
                        status_text.text(f"处理章节: {chapter_title} - 获取到角色: {', '.join(chapter_roles)}")
                    # 保存角色信息到文件
                    roles_file = os.path.join(self.text_segments_dir, f"{safe_title}_roles.txt")
                    with open(roles_file, "w", encoding="utf-8") as f:
                        json.dump(chapter_roles, f, ensure_ascii=False, indent=2)
                else:
                    if status_text:
                        status_text.text(f"处理章节: {chapter_title} - 无法从响应中提取角色信息")
            else:
                if status_text:
                    status_text.text(f"处理章节: {chapter_title} - 角色信息获取失败")
            
            # 第二步：按段落或字数分割
            segments = self.splitter.split_by_paragraph_or_length(chapter_content, self.max_segment_length)
            chapter_json = []
            
            # 保存分割后的片段
            segment_dir = os.path.join(self.text_segments_dir, safe_title)
            os.makedirs(segment_dir, exist_ok=True)
            for i, segment in enumerate(segments):
                segment_file = os.path.join(segment_dir, f"segment_{i+1}.txt")
                with open(segment_file, "w", encoding="utf-8") as f:
                    f.write(segment)
            
            # 处理每个片段
            if status_text:
                status_text.text(f"处理章节: {chapter_title} - 开始处理{len(segments)}个片段")
                
            # 创建进度条和状态文本
            segment_progress_bar = st.progress(0)
            segment_status = st.empty()
            
            # 使用多线程处理片段
            segment_results = [None] * len(segments)
            active_workers = min(self.max_workers, len(segments))
            
            # 创建线程池
            with ThreadPoolExecutor(max_workers=active_workers) as executor:
                # 提交所有任务
                futures = []
                for i, segment in enumerate(segments):
                    future = executor.submit(
                        self.llm_client.process_segment_with_threadpool,
                        segment, i, chapter_roles
                    )
                    futures.append(future)
                    
                    # 更新进度
                    segment_progress = (i + 1) / len(segments)
                    segment_progress_bar.progress(segment_progress)
                    segment_status.text(f"处理章节: {chapter_title} - 提交片段 {i+1}/{len(segments)} 到线程池")
                    
                # 等待所有任务完成
                for _ in tqdm(range(len(segments)), desc=f"处理章节 {chapter_title}"):
                    segment_id, segment_json = self.llm_client.result_queue.get()
                    if segment_json:
                        segment_results[segment_id] = segment_json
                    
                    # 更新进度
                    completed = sum(1 for r in segment_results if r is not None)
                    segment_progress = completed / len(segments)
                    segment_progress_bar.progress(segment_progress)
                    segment_status.text(f"处理章节: {chapter_title} - 完成片段 {completed}/{len(segments)}")
            
            # 收集所有非None的结果
            for segment_result in segment_results:
                if segment_result:
                    chapter_json.extend(segment_result)
            
            # 保存章节结果
            result[chapter_title] = chapter_json
            
            # 保存JSON文件
            chapter_json_file = os.path.join(self.json_results_dir, f"{safe_title}.json")
            with open(chapter_json_file, "w", encoding="utf-8") as f:
                json.dump(chapter_json, f, ensure_ascii=False, indent=2)
            
            if status_text:
                status_text.text(f"处理章节: {chapter_title} - 章节JSON已保存")
                
            # 更新章节进度
            processed_chapters += 1
            if progress_bar:
                progress_bar.progress(processed_chapters / total_chapters)
        
        return result
    
    def save_combined_result(self, result: Dict[str, List[Dict]]):
        """保存合并后的结果"""
        # 合并所有章节的JSON
        all_json = []
        for chapter_title, chapter_json in result.items():
            all_json.extend(chapter_json)
        
        # 保存完整结果到文本同名文件夹
        combined_file = os.path.join(self.output_dir, "combined.json")
        with open(combined_file, "w", encoding="utf-8") as f:
            json.dump(all_json, f, ensure_ascii=False, indent=2)
        
        return combined_file

# 主函数，运行Streamlit应用
def main():
    # 设置页面标题和布局
    st.set_page_config(page_title="TXT转JSON工具", layout="wide")
    st.title("TXT转JSON工具")
    
    # 初始化转换器
    base_dir = os.path.dirname(os.path.abspath(__file__)) or os.getcwd()
    converter = WebUITxtToJsonConverter(base_dir)
    
    # 创建标签页
    tab1, tab2, tab3 = st.tabs(["文件转换", "配置设置", "关于工具"])
    
    with tab1:
        st.header("文件上传与转换")
        
        # 文件上传
        uploaded_file = st.file_uploader("上传TXT文件", type="txt")
        
        if uploaded_file is not None:
            # 显示文件信息
            st.write(f"已上传文件: {uploaded_file.name}")
            
            # 预览文件内容
            if st.checkbox("预览文件内容"):
                file_content = uploaded_file.getvalue().decode("utf-8")
                st.text_area("文件预览", file_content[:2000] + ("..." if len(file_content) > 2000 else ""), height=200)
            
            # 转换按钮
            if st.button("开始转换", key="convert_button"):
                # 检查是否已设置API密钥
                if not st.session_state.get("api_key", ""):
                    st.error("请先在配置设置中填写API密钥")
                else:
                    try:
                        # 保存上传的文件
                        input_file_path = os.path.join(base_dir, uploaded_file.name)
                        with open(input_file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # 读取文件内容
                        with open(input_file_path, "r", encoding="utf-8") as f:
                            text = f.read()
                        
                        # 设置LLM客户端
                        converter.setup_llm_client(
                            api_key=st.session_state.get("api_key"),
                            api_url=st.session_state.get("api_url", "https://api.deepseek.com/v1/chat/completions")
                        )
                        
                        # 设置并发数和片段大小
                        converter.max_workers = st.session_state.get("max_workers", 5)
                        converter.max_segment_length = st.session_state.get("max_segment_length", 350)
                        
                        # 根据输入文件名更新目录结构
                        converter.update_directory_by_filename(input_file_path)
                        
                        # 设置输出目录
                        output_dir = converter.output_dir  # 使用更新后的目录
                        st.session_state.output_dir = output_dir  # 更新会话状态
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir, exist_ok=True)
                        converter.output_dir = output_dir
                        
                        # 创建进度条
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # 处理文本
                        with st.spinner("正在转换..."):
                            start_time = time.time()
                            result = converter.process_text(text, progress_bar, status_text)
                            
                            # 保存合并结果
                            combined_file = converter.save_combined_result(result)
                            
                            end_time = time.time()
                            
                            # 显示完成信息
                            status_text.text(f"转换完成！耗时: {end_time - start_time:.2f}秒")
                            st.success(f"转换成功！")
                            st.info(f"合并后的JSON文件已保存到: {combined_file}")
                            
                            # 提供下载链接
                            with open(combined_file, "r", encoding="utf-8") as f:
                                json_content = f.read()
                            st.download_button(
                                label="下载合并后的JSON文件",
                                data=json_content,
                                file_name=os.path.basename(combined_file),
                                mime="application/json"
                            )
                        
                    except Exception as e:
                        st.error(f"转换过程中发生错误: {str(e)}")
                        import traceback
                        st.exception(traceback.format_exc())
    
    with tab2:
        st.header("配置设置")
        
        # LLM API配置
        st.subheader("LLM API配置")
        
        # 初始化会话状态
        if "api_key" not in st.session_state:
            st.session_state.api_key = ""
        if "api_url" not in st.session_state:
            st.session_state.api_url = "https://api.deepseek.com/v1/chat/completions"
        if "max_workers" not in st.session_state:
            st.session_state.max_workers = 5
        if "max_segment_length" not in st.session_state:
            st.session_state.max_segment_length = 350
        if "output_dir" not in st.session_state:
            st.session_state.output_dir = base_dir
        
        # API密钥输入
        api_key = st.text_input(
            "API密钥",
            value=st.session_state.api_key,
            type="password",
            help="DeepSeek API密钥"
        )
        
        # API URL输入
        api_url = st.text_input(
            "API URL",
            value=st.session_state.api_url,
            help="DeepSeek API的URL地址"
        )
        
        # 并发设置
        st.subheader("处理设置")
        max_workers = st.slider(
            "最大并发数",
            min_value=1,
            max_value=20,
            value=st.session_state.max_workers,
            help="API请求的最大并发数"
        )
        
        # 片段大小设置
        max_segment_length = st.slider(
            "文本片段最大长度",
            min_value=100,
            max_value=2000,
            value=st.session_state.max_segment_length,
            step=50,
            help="文本分割的最大长度"
        )
        
        # 输出目录设置
        output_dir = st.text_input(
            "输出目录",
            value=st.session_state.output_dir,
            help="JSON结果文件的保存目录"
        )
        
        # 保存配置按钮
        if st.button("保存配置"):
            st.session_state.api_key = api_key
            st.session_state.api_url = api_url
            st.session_state.max_workers = max_workers
            st.session_state.max_segment_length = max_segment_length
            st.session_state.output_dir = output_dir
            st.success("配置已保存！")
    
    with tab3:
        st.header("关于本工具")
        st.write("这是一个用于将TXT文本转换为JSON格式的工具，主要功能包括：")
        st.write("1. 支持配置LLM API参数")
        st.write("2. 上传TXT文件并生成JSON结果")
        st.write("3. 支持多线程API请求并发处理")
        st.write("4. 可设置文本切分片段大小")
        st.write("5. 可自定义JSON结果保存路径")
        st.write("")
        st.write("使用说明：")
        st.write("1. 首先在配置设置中填写DeepSeek API密钥和其他参数")
        st.write("2. 然后在文件转换标签页上传TXT文件并点击开始转换")
        st.write("3. 转换完成后可以下载生成的JSON文件")

if __name__ == "__main__":
    main()