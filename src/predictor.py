import torch
from transformers import CLIPProcessor, CLIPModel

class BatchClassifier:
    def __init__(self, model_name="openai/clip-vit-base-patch32", device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model {model_name} to {self.device}...")
        
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        # 定义标签
        self.labels = ["an anime style illustration", "a realistic photo"]
        self.class_names = ["anime", "realistic"]
        
        # 预计算文本特征 (Pre-compute Text Embeddings)
        self._precompute_text_features()

    def _precompute_text_features(self):
        text_inputs = self.processor(text=self.labels, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            self.text_features = self.model.get_text_features(**text_inputs)
            # 归一化，方便后续计算余弦相似度
            self.text_features = self.text_features / self.text_features.norm(p=2, dim=-1, keepdim=True)

    def get_transform(self):
        # 返回 processor 作为 transform 函数
        return self.processor

    def predict_batch(self, batch_pixel_values):
        """
        输入: batch_pixel_values (Tensor) [batch_size, 3, 224, 224]
        输出: (predicted_indices, probs)
        """
        batch_pixel_values = batch_pixel_values.to(self.device)
        
        with torch.no_grad():
            # 1. 计算图像特征
            image_features = self.model.get_image_features(pixel_values=batch_pixel_values)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            
            # 2. 计算相似度 (Image features @ Text features.T)
            # logit_scale 是 CLIP 用来缩放 logits 的参数
            logit_scale = self.model.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ self.text_features.t()
            
            # 3. 计算概率
            probs = logits_per_image.softmax(dim=1)
            
            # 4. 获取最大概率的索引
            max_probs, indices = torch.max(probs, dim=1)
            
        return indices.cpu().numpy(), max_probs.cpu().numpy()