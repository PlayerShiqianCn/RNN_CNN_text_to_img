import os
import re
import time
import requests
import random
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
import pandas as pd
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class BingImageCrawler:
    """Bing图片爬虫类"""
    
    def __init__(self, save_dir='data', headers=None):
        """
        初始化Bing图片爬虫
        
        参数:
            save_dir (str): 图片保存目录
            headers (dict): 请求头
        """
        self.save_dir = save_dir
        self.headers = headers or {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
        }
        
        # 创建保存目录
        self.images_dir = os.path.join(save_dir, 'images')
        self.text_dir = os.path.join(save_dir, 'text')
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.text_dir, exist_ok=True)
        
        # 元数据
        self.metadata_path = os.path.join(save_dir, 'metadata.csv')
        self.metadata = self._load_metadata()
    
    def _load_metadata(self):
        """加载或创建元数据文件"""
        if os.path.exists(self.metadata_path):
            return pd.read_csv(self.metadata_path)
        else:
            return pd.DataFrame(columns=['id', 'query', 'text_path', 'image_path', 'label'])
    
    def _save_metadata(self):
        """保存元数据"""
        self.metadata.to_csv(self.metadata_path, index=False)
    
    def search_images(self, query, num_images=20, first=1):
        """
        搜索图片
        
        参数:
            query (str): 搜索关键词
            num_images (int): 需要下载的图片数量
            first (int): 起始索引
            
        返回:
            list: 图片URL列表
        """
        image_urls = []
        current_count = 0
        
        while current_count < num_images:
            # 构建URL
            search_url = f"https://www.bing.com/images/search?q={query}&first={first}&count=35"
            
            try:
                # 发送请求
                response = requests.get(search_url, headers=self.headers)
                response.raise_for_status()
                
                # 解析HTML
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # 提取图片URL
                img_tags = soup.find_all('a', class_='iusc')
                
                for img_tag in img_tags:
                    try:
                        m = re.search('murl&quot;:&quot;(.*?)&quot;', img_tag['m'])
                        if m:
                            image_url = m.group(1)
                            image_urls.append(image_url)
                            current_count += 1
                            
                            if current_count >= num_images:
                                break
                    except Exception as e:
                        logging.error(f"提取图片URL时出错: {e}")
                
                # 更新起始索引
                first += 35
                
                # 随机延迟，避免被封
                time.sleep(random.uniform(0.5, 2.0))
                
            except Exception as e:
                logging.error(f"搜索图片时出错: {e}")
                break
        
        return image_urls[:num_images]
    
    def download_images(self, query, num_images=20):
        """
        下载图片
        
        参数:
            query (str): 搜索关键词
            num_images (int): 需要下载的图片数量
            
        返回:
            int: 成功下载的图片数量
        """
        logging.info(f"开始下载关键词 '{query}' 的图片，数量: {num_images}")
        
        # 搜索图片
        image_urls = self.search_images(query, num_images)
        logging.info(f"找到 {len(image_urls)} 个图片URL")
        
        # 下载图片
        success_count = 0
        for i, url in enumerate(image_urls):
            try:
                # 发送请求
                response = requests.get(url, headers=self.headers, timeout=10)
                response.raise_for_status()
                
                # 打开图片
                img = Image.open(BytesIO(response.content))
                
                # 生成文件名
                image_id = f"{query.replace(' ', '_')}_{len(self.metadata) + i + 1}"
                image_filename = f"{image_id}.jpg"
                text_filename = f"{image_id}.txt"
                
                # 保存图片
                image_path = os.path.join(self.images_dir, image_filename)
                img.save(image_path)
                
                # 保存文本描述
                text_path = os.path.join(self.text_dir, text_filename)
                with open(text_path, 'w', encoding='utf-8') as f:
                    f.write(query)
                
                # 更新元数据
                new_row = {
                    'id': image_id,
                    'query': query,
                    'text_path': os.path.join('text', text_filename),
                    'image_path': os.path.join('images', image_filename),
                    'label': ''
                }
                self.metadata = pd.concat([self.metadata, pd.DataFrame([new_row])], ignore_index=True)
                
                success_count += 1
                logging.info(f"成功下载图片 {success_count}/{len(image_urls)}: {image_filename}")
                
                # 随机延迟，避免被封
                time.sleep(random.uniform(0.2, 1.0))
                
            except Exception as e:
                logging.error(f"下载图片时出错 ({url}): {e}")
        
        # 保存元数据
        self._save_metadata()
        
        logging.info(f"下载完成，成功: {success_count}/{len(image_urls)}")
        return success_count

# 使用示例
if __name__ == "__main__":
    crawler = BingImageCrawler(save_dir='data')
    crawler.download_images("cat", num_images=5)
    crawler.download_images("dog", num_images=5)