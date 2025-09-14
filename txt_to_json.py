import os
import json
import re
from typing import Dict, List, Tuple
from text_splitter import TextSplitter
from llm_api import LLMClient
from debug_utils import DebugLogger

class TxtToJsonConverter:
    def __init__(self, input_file: str):
        # 初始化基础目录和相关路径
        self.base_dir = os.path.dirname(input_file) or os.getcwd()
        
        # 获取输入文件的文件名（不含扩展名）作为文本同名文件夹名称
        text_filename = os.path.splitext(os.path.basename(input_file))[0]
        
        # 创建\workshop\文本同名文件夹目录结构
        workshop_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workshop")
        text_specific_dir = os.path.join(workshop_dir, text_filename)
        
        # 创建目录
        self.text_segments_dir = os.path.join(text_specific_dir, "text_segments")
        self.json_results_dir = os.path.join(text_specific_dir, "scripts")
        os.makedirs(self.text_segments_dir, exist_ok=True)
        os.makedirs(self.json_results_dir, exist_ok=True)
        
        # 初始化拆分器、API客户端和调试日志器
        self.splitter = TextSplitter()
        
        # 先初始化debug_logger
        self.debug_logger = DebugLogger(os.path.join(self.base_dir, "debug.log"))
        
        # 创建LLMClient实例时传入debug_logger
        self.llm_client = LLMClient(debug_logger=self.debug_logger)
        
        # 加载prompt模板
        prompt_file = os.path.join(self.base_dir, "json生成prompt")
        chapter_prompt_file = os.path.join(self.base_dir, "章节prompt")
        self.llm_client.load_prompt_template(prompt_file)
        self.llm_client.load_chapter_prompt_template(chapter_prompt_file)

    def process_text(self, text: str) -> Dict[str, List[Dict]]:
        """处理完整文本，返回按章节组织的JSON结果"""
        # 第一步：按章节分割
        chapters = self.splitter.split_by_chapter(text)
        result = {}
        
        for chapter_title, chapter_content in chapters:
            print(f"处理章节: {chapter_title}")
            
            # 保存原始章节文本
            safe_title = re.sub(r'[\\/:*?"<>|]', '_', chapter_title)
            chapter_text_file = os.path.join(self.text_segments_dir, f"{safe_title}.txt")
            with open(chapter_text_file, "w", encoding="utf-8") as f:
                f.write(chapter_content)
            
            # 获取章节角色信息
            print(f"  获取章节角色信息...")
            chapter_roles_response = self.llm_client.call_llm_for_chapter_roles(chapter_content)
            chapter_roles = []
            if chapter_roles_response:
                chapter_roles = self.llm_client.extract_roles_from_response(chapter_roles_response)
                if chapter_roles:
                    print(f"  获取到的角色: {chapter_roles}")
                    # 保存角色信息到文件
                    roles_file = os.path.join(self.text_segments_dir, f"{safe_title}_roles.txt")
                    with open(roles_file, "w", encoding="utf-8") as f:
                        json.dump(chapter_roles, f, ensure_ascii=False, indent=2)
                else:
                    print(f"  无法从响应中提取角色信息")
            else:
                print(f"  角色信息获取失败")
            
            # 第二步：按段落或字数分割
            segments = self.splitter.split_by_paragraph_or_length(chapter_content)
            chapter_json = []
            
            # 保存分割后的片段
            segment_dir = os.path.join(self.text_segments_dir, safe_title)
            os.makedirs(segment_dir, exist_ok=True)
            for i, segment in enumerate(segments):
                segment_file = os.path.join(segment_dir, f"segment_{i+1}.txt")
                with open(segment_file, "w", encoding="utf-8") as f:
                    f.write(segment)
            
            # 处理每个片段
            for i, segment in enumerate(segments):
                print(f"  处理片段 {i+1}/{len(segments)}")
                
                # 准备片段处理的完整提示
                segment_with_roles = segment
                if chapter_roles:
                    roles_text = ",".join(chapter_roles)
                    segment_with_roles = f"从以下角色中选取这段文本的角色，请注意这段文本只是章节片段，不是所有角色一定都要出现。角色列表：{roles_text}\n\n{segment}"
                
                # 调用LLM处理片段
                llm_response = self.llm_client.call_llm(segment_with_roles)
                if not llm_response:
                    print(f"  片段 {i+1} 处理失败，跳过")
                    continue
                
                # 提取JSON
                segment_json = self.llm_client.extract_json_from_response(llm_response)
                if segment_json:
                    chapter_json.extend(segment_json)
                else:
                    print(f"  无法从响应中提取JSON，跳过片段 {i+1}")
            
            # 保存章节结果（每生成完一章就保存）
            result[chapter_title] = chapter_json
            
            # 保存JSON文件
            chapter_json_file = os.path.join(self.json_results_dir, f"{safe_title}.json")
            with open(chapter_json_file, "w", encoding="utf-8") as f:
                json.dump(chapter_json, f, ensure_ascii=False, indent=2)
            print(f"  章节JSON已保存到: {chapter_json_file}")
        
        return result
    
    def save_combined_result(self, result: Dict[str, List[Dict]]):
        """保存合并后的结果"""
        # 合并所有章节的JSON
        all_json = []
        for chapter_title, chapter_json in result.items():
            all_json.extend(chapter_json)
        
        # 保存完整结果到文本同名文件夹
        combined_file = os.path.join(os.path.dirname(self.text_segments_dir), f"combined.json")
        with open(combined_file, "w", encoding="utf-8") as f:
            json.dump(all_json, f, ensure_ascii=False, indent=2)
        
        print(f"合并后的JSON已保存到: {combined_file}")

import sys

def main():
    """主函数"""
    # 尝试从命令行参数获取文件路径
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        # 尝试自动检测当前目录中的TXT文件
        txt_files = [f for f in os.listdir('.') if f.endswith('.txt') and not f.startswith('.')]
        if txt_files:
            # 优先选择看起来像小说的文件（排除requirements.txt等配置文件）
            non_config_files = [f for f in txt_files if not f.lower() in ['requirements.txt', 'readme.txt', 'license.txt']]
            if non_config_files:
                input_file = non_config_files[0]
            else:
                # 如果没有非配置文件，就选择第一个txt文件
                input_file = txt_files[0]
            print(f"自动选择文件: {input_file}")
        else:
            # 否则提示用户输入
            input_file = input("请输入要处理的TXT文件路径: ")
    
    # 读取输入文件
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception as e:
        print(f"读取文件错误: {e}")
        return
    
    # 创建转换器并处理文本
    try:
        converter = TxtToJsonConverter(input_file)
        result = converter.process_text(text)
        
        # 保存合并后的结果
        converter.save_combined_result(result)
    except ValueError as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    main()