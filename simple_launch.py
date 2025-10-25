#!/usr/bin/env python3
"""
简单的GUI测试脚本
"""

if __name__ == "__main__":
    print("🚀 启动 LLM-Feynman GUI...")
    
    try:
        import gradio as gr
        print("✅ Gradio已安装，版本:", gr.__version__)
        
        # 导入GUI
        from gradio_gui import create_gui
        
        # 创建界面
        interface = create_gui()
        print("✅ GUI创建成功")
        
        # 启动界面
        print("🌐 启动Web界面，访问地址: http://127.0.0.1:7860")
        print("📝 按 Ctrl+C 停止服务")
        
        interface.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=False,
            show_error=True,
            quiet=False
        )
        
    except KeyboardInterrupt:
        print("\n👋 GUI已停止")
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        import traceback
        traceback.print_exc()