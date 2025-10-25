# LLM-Feynman GUI 使用指南

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements_gui.txt
```

### 2. 启动GUI

```bash
python launch_gui.py
```

然后在浏览器中访问 `http://localhost:7860`

## 📋 使用流程

### 步骤1: 数据上传
- 支持CSV和Excel格式文件
- 数据应包含数值特征列和一个目标列
- 建议数据量 > 20行

### 步骤2: 选择目标列
- 从下拉菜单中选择要预测的目标变量
- 其他列自动作为特征列

### 步骤3: 配置物理含义
- **目标变量含义**: 描述目标变量的物理意义
- **目标变量量纲**: 指定单位（如：V, K, mol/s等）
- **研究背景**: 描述研究目标和期望发现的规律
- **特征含义**: JSON格式描述每个特征的物理含义
- **特征量纲**: JSON格式指定每个特征的单位

### 步骤4: 模型配置
- **模型类型**: 选择huggingface、openai或o3
- **O3平台**: 支持Claude Sonnet 4、GPT-4o等高性能模型
- **API配置**: 根据选择的模型类型配置相应的API密钥
- **算法参数**: 调整公式发现算法的参数

#### O3平台模型支持
- **Claude Sonnet 4 (推荐)**: claude-sonnet-4-20250514
- **GPT-4o**: gpt-4o  
- **GPT-4o Mini**: gpt-4o-mini
- **O1 Preview**: o1-preview
- **O1 Mini**: o1-mini

#### 模型配置示例
```
模型类型: o3
模型名称: claude-sonnet-4-20250514
API Key: sk-wqy8ahJLPeT4cTvmCbDcD1B9AeBf4fAcA610649b882fE5Fd
Base URL: https://api.o3.fan/v1
```

### 步骤5: 运行发现
- 点击"开始发现公式"按钮
- 等待算法完成，查看发现的公式
- 下载结果进行进一步分析

## 📊 测试数据

项目包含一个测试数据文件 `test_data_ohms_law.csv`，包含电流、电阻和电压数据，用于验证欧姆定律 (V = I × R)。

### 测试配置示例

**目标变量**: voltage
**目标含义**: Electric voltage across the component  
**目标量纲**: V

**特征含义**:
```json
{
  "current": "Electric current flowing through the circuit",
  "resistance": "Electrical resistance of the component"
}
```

**特征量纲**:
```json
{
  "current": "A",
  "resistance": "Ω"
}
```

**研究背景**: "We are investigating the fundamental relationship between electrical current, resistance, and voltage in DC circuits. This is a classic physics problem where we expect to discover Ohm's Law (V = I × R)."

## 🎯 最佳实践

### 数据准备
- 确保数据质量良好，无过多缺失值
- 特征和目标变量应为数值型
- 建议进行基本的数据清理

### 物理描述
- 详细描述变量的物理含义
- 正确指定量纲单位
- 提供清晰的研究背景

### 参数调优
- 从较小的参数开始测试
- 根据数据复杂度调整公式数量
- 复杂数据可增加迭代次数

## 🔧 高级功能

### 🤖 AI智能模板生成
使用大语言模型自动生成物理含义和量纲模板：
- 点击"🤖 AI智能生成"按钮
- 系统会根据变量名称和研究背景自动推测物理含义
- 自动生成标准的物理量纲
- 支持中文物理含义描述

### 量纲分析
启用量纲分析可以确保发现的公式在物理上是合理的。

### 公式解释
包含公式解释选项会使用MCTS算法生成公式的物理解释。

### 结果导出
- **JSON格式**: 包含完整的公式信息和元数据
- **CSV格式**: 简化的表格格式，便于Excel分析

## ⚠️ 注意事项

- 首次运行可能需要下载模型，时间较长
- 某些大型模型需要GPU支持
- 运行时间取决于数据大小和参数设置
- 建议在有足够内存的环境中运行

## 🐛 故障排除

### 常见问题

1. **模块导入错误**
   - 检查Python路径设置
   - 确保在正确的目录运行

2. **模型加载失败**
   - 检查网络连接
   - 验证模型名称是否正确

3. **内存不足**
   - 减少初始公式数量
   - 使用较小的模型

4. **结果质量不佳**
   - 检查数据质量
   - 调整算法参数
   - 改进物理描述

## 📈 结果解读

### 性能指标
- **R² 得分**: 拟合优度，越接近1越好
- **MAE**: 平均绝对误差，越小越好
- **复杂度**: 公式复杂程度，平衡准确性和可解释性
- **可解释性**: 公式的物理可解释程度

### 公式选择
- 优先选择R²高且复杂度适中的公式
- 考虑物理可解释性
- 验证量纲的正确性

## 📞 技术支持

如遇问题，请：
1. 检查错误信息和日志
2. 参考项目README
3. 在GitHub Issues中报告问题