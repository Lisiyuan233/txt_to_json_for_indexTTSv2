import re
from typing import List, Tuple

class TextSplitter:
    def __init__(self):
        """初始化文本分割器"""
        pass
    
    def split_by_chapter(self, text: str) -> List[Tuple[str, str]]:
        """按章节分割文本"""
        chapters = []
        chapter_index = 1  # 添加章节索引计数器
        
        # 首先检查是否有#开头的行作为章节
        hashtag_chapters = []
        lines = text.split('\n')
        current_content = []
        found_hashtag = False
        
        for line in lines:
            if '#' in line.strip():
                found_hashtag = True
                # 如果之前有内容，保存为一个章节
                if current_content:
                    content = '\n'.join(current_content).strip()
                    if content:
                        hashtag_chapters.append((f"第{chapter_index}章", content))
                        chapter_index += 1
                        current_content = []
            else:
                current_content.append(line)
        
        # 处理最后一段内容
        if current_content:
            content = '\n'.join(current_content).strip()
            if content:
                hashtag_chapters.append((f"第{chapter_index}章", content))
        
        # 如果找到了#开头的章节，直接返回这些章节
        if found_hashtag and hashtag_chapters:
            return hashtag_chapters
        
        # 如果没有#开头的章节，则使用原来的其他章节匹配方法
        # 使用正则表达式匹配章节标题
        # 常见的章节格式如：第一章、第1章、第一回、Chapter 1等
        chapter_patterns = [
            r'(第[一二三四五六七八九十百千]+章)',  # 中文数字章节
            r'(第\d+章)',                          # 阿拉伯数字章节
            r'(第[一二三四五六七八九十百千]+回)',  # 中文数字回目
            r'(第\d+回)',                          # 阿拉伯数字回目
            r'(Chapter\s+\d+)',                    # 英文章节
            r'(CHAPTER\s+\d+)',                    # 大写英文章节
        ]
        
        chapters = []
        start_pos = 0
        chapter_index = 1  # 重置章节索引计数器
        
        # 合并所有模式进行搜索
        combined_pattern = '|'.join(chapter_patterns)
        
        # 首先检查文本开头是否有内容（非章节标题）
        first_chapter_match = re.search(combined_pattern, text)
        if first_chapter_match and first_chapter_match.start() > 0:
            # 文本开头有内容，将其作为第一章
            first_content = text[:first_chapter_match.start()].strip()
            if first_content:
                chapters.append((f"第{chapter_index}章", first_content))
                chapter_index += 1
                start_pos = first_chapter_match.start()
        
        # 处理其余章节
        for match in re.finditer(combined_pattern, text[start_pos:]):
            chapter_title = match.group(0)
            chapter_start = start_pos + match.start()
            chapter_end = start_pos + match.end()
            
            # 提取章节内容（从当前章节标题到下一个章节标题之前）
            next_chapter_match = re.search(combined_pattern, text[chapter_end:])
            if next_chapter_match:
                chapter_content = text[chapter_end:chapter_end + next_chapter_match.start()].strip()
            else:
                chapter_content = text[chapter_end:].strip()
            
            # 如果章节内容不为空，则添加到结果中
            if chapter_content:
                chapters.append((f"第{chapter_index}章", chapter_content))
                chapter_index += 1
        
        # 如果没有检测到章节，将整个文本作为第一章
        if not chapters and text.strip():
            chapters.append(("第1章", text.strip()))
            
        return chapters
    
    def split_by_paragraph_or_length(self, text: str, max_length: int = 350) -> List[str]:
        """按标点符号分割文本，确保每个片段由标点分隔，且完整句子不会被分割到不同片段"""
        result = []
        text = text.strip()
        
        if not text:
            return result
        
        # 定义句子结束标点（优先级高）和其他标点（优先级低）
        sentence_end_punctuations = ["。", "！", "？", ".", "!", "?"]
        other_punctuations = ["，", ",", "；", ";", "：", ":", "、"]
        all_punctuations = sentence_end_punctuations + other_punctuations
        
        # 引号对类型
        quote_pairs = [("'", "'"), ("'", "'"), ("'", "'"), ("'", "'")]
        
        start = 0
        while start < len(text):
            # 设置初始结束位置
            end = min(start + max_length, len(text))
            
            # 检查是否有未闭合的引号，如果有则扩展end到引号闭合位置
            # 查找当前范围内所有引号的位置
            quotes_positions = []
            for open_quote, close_quote in quote_pairs:
                for match in re.finditer(re.escape(open_quote), text[start:end]):
                    quotes_positions.append((match.start() + start, 'open', open_quote, close_quote))
                for match in re.finditer(re.escape(close_quote), text[start:end]):
                    quotes_positions.append((match.start() + start, 'close', open_quote, close_quote))
            
            # 按位置排序
            quotes_positions.sort(key=lambda x: x[0])
            
            # 检查是否有未闭合的引号
            quote_stack = []
            for pos, type_, open_q, close_q in quotes_positions:
                if type_ == 'open':
                    quote_stack.append((open_q, close_q, pos))
                else:
                    # 寻找匹配的开引号
                    matched = False
                    for i in range(len(quote_stack)-1, -1, -1):
                        stack_open, stack_close, _ = quote_stack[i]
                        if stack_close == close_q:
                            quote_stack.pop(i)
                            matched = True
                            break
                    # 如果没有匹配的开引号，可能是单引号作为缩写等特殊情况，忽略
            
            # 如果有未闭合的引号，扩展end直到所有引号闭合
            while quote_stack and end < len(text):
                # 继续查找闭合引号
                remaining_text = text[end:]
                all_found = True
                
                for open_q, close_q, _ in quote_stack:
                    # 查找下一个闭合引号
                    close_pos = remaining_text.find(close_q)
                    if close_pos != -1:
                        # 更新end到闭合引号之后
                        end = end + close_pos + 1
                    else:
                        all_found = False
                        break
                
                if not all_found:
                    # 如果没有找到所有闭合引号，尝试找下一组引号
                    break
            
            # 确保不会强制分割句子，优先使用句子结束标点
            best_split_pos = -1
            
            # 1. 优先查找句子结束标点
            for punctuation in sentence_end_punctuations:
                pos = text.rfind(punctuation, start, end)
                if pos > start:
                    best_split_pos = max(best_split_pos, pos)
            
            # 2. 如果没有找到句子结束标点，查找其他标点
            if best_split_pos == -1:
                for punctuation in other_punctuations:
                    pos = text.rfind(punctuation, start, end)
                    if pos > start:
                        best_split_pos = max(best_split_pos, pos)
            
            # 3. 如果仍然没有找到标点，并且文本长度超过最大长度的90%，尝试扩展查找范围
            if best_split_pos == -1 and end - start >= max_length * 0.9:
                # 扩展查找范围继续寻找标点
                extended_end = min(start + int(max_length * 1.2), len(text))
                for punctuation in all_punctuations:
                    pos = text.rfind(punctuation, start, extended_end)
                    if pos > start:
                        best_split_pos = max(best_split_pos, pos)
                end = extended_end
            
            # 4. 如果找到了合适的标点符号，就在此分割
            if best_split_pos > start:
                result.append(text[start:best_split_pos+1])
                start = best_split_pos + 1
            else:
                # 最后万不得已的情况，只能按最大长度分割，但这应该很少发生
                result.append(text[start:end])
                start = end
        
        # 清理空字符串
        result = [segment.strip() for segment in result if segment.strip()]
        
        # 再次检查并确保每个片段的完整性和句子连贯性
        final_result = []
        for segment in result:
            if len(segment) <= max_length * 1.5:  # 允许一定的长度弹性
                final_result.append(segment)
            else:
                # 对于超长片段，进行更细致的分割，但严格按照标点符号
                sub_start = 0
                while sub_start < len(segment):
                    sub_end = min(sub_start + max_length, len(segment))
                    
                    # 检查子片段中是否有未闭合的引号
                    for open_quote, close_quote in quote_pairs:
                        open_count = segment[sub_start:sub_end].count(open_quote)
                        close_count = segment[sub_start:sub_end].count(close_quote)
                        if open_count > close_count:
                            # 查找子片段后的闭合引号
                            remaining_in_segment = segment[sub_end:]
                            close_pos = remaining_in_segment.find(close_quote)
                            if close_pos != -1:
                                sub_end = sub_end + close_pos + 1
                    
                    # 在子片段中寻找最佳分割点（标点符号）
                    sub_best_split = -1
                    for punctuation in all_punctuations:
                        pos = segment.rfind(punctuation, sub_start, sub_end)
                        if pos > sub_start:
                            sub_best_split = max(sub_best_split, pos)
                    
                    if sub_best_split > sub_start:
                        final_result.append(segment[sub_start:sub_best_split+1])
                        sub_start = sub_best_split + 1
                    else:
                        # 最后实在没有标点，才强制分割
                        final_result.append(segment[sub_start:sub_end])
                        sub_start = sub_end
        
        return final_result