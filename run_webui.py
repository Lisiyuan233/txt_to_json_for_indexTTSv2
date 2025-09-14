import os
import sys
import subprocess

# 获取当前脚本所在目录
base_dir = os.path.dirname(os.path.abspath(__file__))

# 检查webui.py是否存在
webui_path = os.path.join(base_dir, "webui.py")
if not os.path.exists(webui_path):
    print(f"错误: 找不到webui.py文件在 {base_dir}")
    sys.exit(1)

# 检查Python环境
python_exe = sys.executable
print(f"使用Python解释器: {python_exe}")

# 检查是否安装了所需依赖
try:
    import streamlit
    import tqdm
    import requests
    print("所有必需的依赖包已安装")
except ImportError as e:
    print(f"缺少依赖包: {e}")
    print("尝试安装依赖包...")
    try:
        subprocess.run(
            [python_exe, "-m", "pip", "install", "-r", os.path.join(base_dir, "requirements.txt")],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print("依赖包安装成功")
    except subprocess.CalledProcessError as ce:
        print(f"依赖包安装失败: {ce}")
        print("请手动安装依赖: pip install -r requirements.txt")
        sys.exit(1)

# 启动Streamlit WebUI
print("正在启动TXT转JSON工具Web界面...")
try:
    # 运行Streamlit命令，使用更高的端口8888
    subprocess.run(
        [python_exe, "-m", "streamlit", "run", webui_path, "--server.port", "8888", "--server.maxUploadSize", "500"],
        cwd=base_dir
    )
except KeyboardInterrupt:
    print("用户中断程序")
except Exception as e:
    print(f"启动WebUI失败: {e}")
    print("请尝试手动运行: streamlit run webui.py")
    sys.exit(1)