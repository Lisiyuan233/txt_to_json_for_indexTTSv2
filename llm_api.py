import os
import time
import requests
import json
import re
from debug_utils import DebugLogger

class LLMClient:
    def __init__(self, api_key=None, api_url="https://api.deepseek.com/v1/chat/completions", debug_logger=None):
        """初始化LLM客户端，设置API密钥和URL"""
        # 从环境变量获取API密钥
        self.api_key = api_key if api_key else os.environ.get("DEEPSEEK-API-KEY")
        if not self.api_key:
            raise ValueError("API密钥未设置，请先设置DEEPSEEK-API-KEY环境变量或直接提供api_key参数")
            
        self.api_url = api_url
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # 加载prompt模板（这里可以通过参数传入或从文件加载）
        self.prompt_template = None
        self.chapter_prompt_template = None
        
        # 初始化debug日志记录器
        self.debug_logger = debug_logger
    
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
    
    def call_llm(self, text: str) -> str:
        """调用DeepSeek API处理文本"""
        if not self.prompt_template:
            raise ValueError("prompt模板未加载，请先调用load_prompt_template方法")
        
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
        
        try:
            # 如果有debug_logger，写入请求内容
            if self.debug_logger:
                self.debug_logger.write_debuglog("JSON生成请求", text, "待发送")
                
            response = requests.post(self.api_url, headers=self.headers, json=data, timeout=60)
            response.raise_for_status()
            
            # 解析响应
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                llm_response = result["choices"][0]["message"]["content"]
                # 如果有debug_logger，写入响应内容
                if self.debug_logger:
                    self.debug_logger.write_debuglog("JSON生成响应", text, llm_response)
                return llm_response
            else:
                error_msg = "API response format error"
                if self.debug_logger:
                    self.debug_logger.write_debuglog("JSON生成响应", text, error_msg)
                raise Exception(error_msg)
                
        except Exception as e:
            print(f"API调用错误: {e}")
            # 重试一次
            time.sleep(2)
            try:
                response = requests.post(self.api_url, headers=self.headers, json=data, timeout=60)
                response.raise_for_status()
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    llm_response = result["choices"][0]["message"]["content"]
                    # 如果有debug_logger，写入重试响应内容
                    if self.debug_logger:
                        self.debug_logger.write_debuglog("JSON生成响应(重试)", text, llm_response)
                    return llm_response
                else:
                    error_msg = "API response format error after retry"
                    if self.debug_logger:
                        self.debug_logger.write_debuglog("JSON生成响应(重试)", text, error_msg)
                    raise Exception(error_msg)
            except Exception as e2:
                print(f"API重试失败: {e2}")
                if self.debug_logger:
                    self.debug_logger.write_debuglog("JSON生成响应(重试失败)", text, str(e2))
                return ""
    
    def call_llm_for_chapter_roles(self, text: str) -> str:
        """调用DeepSeek API获取章节角色信息"""
        if not self.chapter_prompt_template:
            raise ValueError("章节prompt模板未加载，请先调用load_chapter_prompt_template方法")
        
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
        
        try:
            # 如果有debug_logger，写入请求内容
            if self.debug_logger:
                self.debug_logger.write_debuglog("章节角色信息请求", text, "待发送")
                
            response = requests.post(self.api_url, headers=self.headers, json=data, timeout=60)
            response.raise_for_status()
            
            # 解析响应
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                llm_response = result["choices"][0]["message"]["content"]
                # 如果有debug_logger，写入响应内容
                if self.debug_logger:
                    self.debug_logger.write_debuglog("章节角色信息响应", text, llm_response)
                return llm_response
            else:
                error_msg = "API response format error"
                if self.debug_logger:
                    self.debug_logger.write_debuglog("章节角色信息响应", text, error_msg)
                raise Exception(error_msg)
                
        except Exception as e:
            print(f"获取章节角色信息错误: {e}")
            # 重试一次
            time.sleep(2)
            try:
                response = requests.post(self.api_url, headers=self.headers, json=data, timeout=60)
                response.raise_for_status()
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    llm_response = result["choices"][0]["message"]["content"]
                    # 如果有debug_logger，写入重试响应内容
                    if self.debug_logger:
                        self.debug_logger.write_debuglog("章节角色信息响应(重试)", text, llm_response)
                    return llm_response
                else:
                    error_msg = "API response format error after retry"
                    if self.debug_logger:
                        self.debug_logger.write_debuglog("章节角色信息响应(重试)", text, error_msg)
                    raise Exception(error_msg)
            except Exception as e2:
                print(f"获取章节角色信息重试失败: {e2}")
                if self.debug_logger:
                    self.debug_logger.write_debuglog("章节角色信息响应(重试失败)", text, str(e2))
                return ""
    
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
                print("角色响应格式错误，不是列表")
                return []
        except json.JSONDecodeError:
            # 如果失败，尝试提取JSON代码块
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
            if json_match:
                try:
                    roles = json.loads(json_match.group(1))
                    if isinstance(roles, list):
                        return roles
                    else:
                        print("提取的角色格式错误，不是列表")
                        return []
                except json.JSONDecodeError:
                    print("无法解析提取的角色JSON")
                    return []
            else:
                print("响应中未找到角色JSON")
                return []