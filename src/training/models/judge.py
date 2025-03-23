import torch
from torch import nn
from typing import List
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration


class SceneryJudge(nn.Module):
    """
    SceneryJudge是一个文本分类器，用于评估场景问答的准确性。
    """
    def __init__(self, pretrained_model="Qwen/Qwen2-VL-2B-Instruct"):
        super().__init__()
        self.tokenizer = AutoProcessor.from_pretrained(pretrained_model)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            pretrained_model
        ).eval()

    @torch.inference_mode()
    def forward(self, question: str, references: List[str], prediction: str):
        """
        评估模型的推理函数，通过比较生成文本与参考答案的相似度评分。
        Args:
            question: 输入问题。
            references: 参考答案列表。
            prediction: 模型预测。
        Output:
            scores: 指示准确性的分数。
        """
        device = next(self.parameters()).device
        texts = [
            f"问题: {question}\n参考答案: {a_gt}\n预测: {prediction}"
            for a_gt in references
        ]

        # 使用生成模型计算困惑度或相似度作为评分
        scores = []
        for text in texts:
            encoded_input = self.tokenizer.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
            encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
            with torch.no_grad():
                outputs = self.model(**encoded_input, labels=encoded_input["input_ids"])
                # 使用模型的损失作为评分指标（损失越低，评分越高）
                loss = outputs.loss
                # 转换为分数（损失越低，分数越高）
                score = torch.exp(-loss)
            scores.append(score.item())
        return torch.tensor(scores)

    def compute(self, questions: List[str], references: List[List[str]], predictions: List[str]):
        """
        计算评估指标。对于多个参考答案，选择最高分数。
        Args:
            questions: 输入问题列表。
            references: 参考答案列表的列表，每个问题支持多个参考答案。
            predictions: 模型预测列表。
        Output:
            scores: 指示准确性的分数。
        """
        max_scores = []

        for index, question in enumerate(questions):
            references_preprocessed = [self.preprocess(reference) for reference in references[index]]
            prediction_preprocessed = self.preprocess(predictions[index])
            scores = self.forward(question, references_preprocessed, prediction_preprocessed)
            max_score = [max(scores)]  # 包装在列表中
            max_scores.extend(max_score)  # 使用extend而不是append
        return torch.tensor(max_scores)
        
    def preprocess(self, string: str):
        """
        一致性预处理函数。
        Args:
            string: 要处理的输入字符串。
        Output:
            output: 处理后的字符串，小写并去除前后空格。
        """
        output = str(string).lower().lstrip().rstrip() 
        return output 