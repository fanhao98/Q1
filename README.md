# 高收益量化交易系统 V12.0 Pro

基于 Tushare 真实股票数据的量化分析与交易策略系统。

## 项目简介

本项目是一个功能完整的股票量化交易系统，集成了实时行情数据获取、K 线图表展示、技术指标分析、策略回测与参数优化等功能。系统采用 Python Flask 作为后端服务，纯 HTML/CSS/JavaScript 作为前端界面。

## 主要功能

### 数据服务
- **股票列表获取**: 获取 A 股全市场股票基本信息
- **概念板块数据**: 获取同花顺概念板块列表及成分股
- **指数成分股**: 获取主流指数（上证指数、深证成指、创业板指等）成分股
- **K 线数据**: 获取个股历史日 K 线数据，支持前复权

### 技术分析
- **K 线图展示**: 使用 ECharts 绘制专业 K 线图
- **成交量分析**: 同步展示成交量柱状图
- **多周期支持**: 支持日线、周线、月线数据查看

### 量化策略
- **策略回测**: 基于历史数据验证交易策略有效性
- **参数优化**: 自动寻找最佳策略参数组合
- **收益计算**: 计算策略收益率、最大回撤、夏普比率等指标

### 用户功能
- **股票池管理**: 自定义关注股票列表
- **配置导入导出**: 支持策略配置的备份与恢复
- **主题切换**: 支持深色/浅色主题
- **本地状态保存**: 自动保存用户界面状态

## 技术架构

```
┌─────────────────────────────────────────────────────────┐
│                      前端层                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ quant_frontend│  │  help.html   │  │echarts.min.js│  │
│  │   .html      │  │   (帮助页)    │  │  (图表库)    │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
                           │ HTTP REST API
┌─────────────────────────────────────────────────────────┐
│                      后端层 (Flask)                      │
│  ┌──────────────────────────────────────────────────┐  │
│  │              quant_server.py                      │  │
│  │  • 股票数据 API    • 策略计算引擎   • 配置管理    │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                           │
┌─────────────────────────────────────────────────────────┐
│                      数据源                              │
│              Tushare Pro API (财经数据接口)              │
└─────────────────────────────────────────────────────────┘
```

## 项目结构

```
.
├── quant_server.py          # 主程序 - Flask 后端服务
├── quant_frontend.html      # 前端主页面
├── help.html                # 使用帮助页面
├── app_launcher.py          # 应用启动器（自动打开浏览器）
├── code_generator.py        # Claude 代码生成器工具
├── api/
│   └── index.py             # Vercel Serverless API 适配
├── config/
│   └── user_config.json     # 用户配置文件
├── local_state/             # 本地状态数据目录
├── concept_members/         # 概念板块成分股缓存
├── requirements.txt         # Python 依赖列表
├── vercel.json              # Vercel 部署配置
├── build_exe.ps1            # Windows EXE 打包脚本
├── TushareQuantSystem.spec  # PyInstaller 打包配置
└── PACKAGING_WINDOWS.md     # Windows 打包说明
```

## 安装与运行

### 环境要求
- Python 3.9+
- Tushare Pro Token（[免费申请](https://tushare.pro/register)）

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

依赖列表：
- flask - Web 框架
- flask-cors - 跨域支持
- tushare - 财经数据接口
- pandas - 数据处理
- numpy - 数值计算

### 2. 配置 Tushare Token

方式一：设置环境变量（推荐）
```bash
# Windows
set TUSHARE_TOKEN=your_token_here

# Linux/Mac
export TUSHARE_TOKEN=your_token_here
```

方式二：创建 token 文件
在 `%LOCALAPPDATA%\TushareQuantSystem\tushare_token.txt` 文件中写入 token。

### 3. 启动服务

```bash
# 方式一：直接启动（需手动打开浏览器）
python quant_server.py

# 方式二：使用启动器（自动打开浏览器）
python app_launcher.py
```

服务启动后访问：http://127.0.0.1:5000

## API 接口列表

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/health` | GET | 服务健康检查 |
| `/api/stock_list` | GET | 获取股票列表 |
| `/api/concept_list` | GET | 获取概念板块列表 |
| `/api/concept_members` | GET | 获取概念成分股 |
| `/api/index_stocks` | GET | 获取指数成分股 |
| `/api/stock_data` | GET | 获取个股 K 线数据 |
| `/api/best_strategy` | POST | 执行策略回测 |
| `/api/auto_optimize_params` | POST | 启动参数优化 |
| `/api/auto_optimize_progress` | GET | 获取优化进度 |
| `/api/config` | GET/POST | 配置读写 |
| `/api/config/export` | GET | 导出配置 |
| `/api/config/import` | POST | 导入配置 |
| `/api/local_state` | GET/POST | 本地状态管理 |

## 部署方式

### 本地运行
适合个人使用，数据存储在本地。

```bash
python app_launcher.py
```

### Vercel 部署
支持部署到 Vercel 作为在线服务。

```bash
# 安装 Vercel CLI
npm i -g vercel

# 部署
vercel
```

### Windows EXE 打包
使用 PyInstaller 打包成独立可执行文件。

```powershell
powershell -ExecutionPolicy Bypass -File .\build_exe.ps1
```

输出目录：`dist\TushareQuantSystem\`

详细说明见 [PACKAGING_WINDOWS.md](PACKAGING_WINDOWS.md)

## 使用说明

1. **首次使用**: 配置 Tushare Token 后启动服务
2. **浏览行情**: 在搜索框输入股票代码或名称查看 K 线
3. **添加自选**: 将关注的股票加入股票池
4. **策略回测**: 选择策略参数，点击"运行回测"查看结果
5. **参数优化**: 使用自动优化功能寻找最佳参数组合

详细帮助请查看系统内的"使用帮助"页面或 [help.html](help.html)。

## 注意事项

1. **Token 安全**: 不要将 Tushare Token 提交到代码仓库
2. **数据缓存**: 系统会自动缓存股票列表等数据，减少 API 调用
3. **频率限制**: Tushare 免费版有 API 调用频率限制，请合理使用
4. **免责声明**: 本系统仅供学习研究使用，不构成投资建议

## 许可证

MIT License

## 致谢

- [Tushare](https://tushare.pro/) - 提供财经数据接口
- [ECharts](https://echarts.apache.org/) - 提供图表库
- [Flask](https://flask.palletsprojects.com/) - Web 框架
