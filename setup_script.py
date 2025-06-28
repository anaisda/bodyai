#!/usr/bin/env python3
"""
Automated Setup Script for AI Body Measurement Web System
=========================================================

This script will automatically set up your project for web deployment.
Run this in your project directory to prepare everything for hosting.
"""

import os
import sys
import shutil
from pathlib import Path
import subprocess

def create_file(path, content):
    """Create a file with the given content"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✅ Created: {path}")

def setup_project():
    """Set up the complete project structure"""
    
    print("🚀 Setting up AI Body Measurement Web System...")
    print("=" * 60)
    
    # Create directory structure
    directories = [
        'uploads',
        'results', 
        'logs',
        'src'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"📁 Created directory: {directory}")
    
    # Create requirements.txt
    requirements = """Flask==2.3.3
Flask-CORS==4.0.0
opencv-python-headless==4.8.1.78
numpy==1.24.3
Pillow==10.0.1
gunicorn==21.2.0
werkzeug==2.3.7
mediapipe==0.10.7
scipy==1.11.3
ultralytics==8.0.196
torch==2.0.1+cpu
torchvision==0.15.2+cpu
psutil==5.9.6
pyyaml==6.0.1"""
    
    create_file('requirements.txt', requirements)
    
    # Create Procfile for Heroku
    create_file('Procfile', 'web: gunicorn app:app')
    
    # Create runtime.txt for Heroku
    create_file('runtime.txt', 'python-3.9.18')
    
    # Create .gitignore
    gitignore = """__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Application specific
uploads/
results/
logs/
*.log

# Model files
models/
*.pt
*.pth
*.onnx

# Environment variables
.env
.env.local"""
    
    create_file('.gitignore', gitignore)
    
    # Create __init__.py files
    create_file('src/__init__.py', '# AI Body Measurement System')
    
    # Create simple README
    readme = """# AI Body Measurement System - Web Edition

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run locally:
   ```bash
   python app.py
   ```

3. Open browser to: http://localhost:5000

## Deployment

Follow the setup guide for deploying to Railway, Render, or Heroku.

## Features

- Multi-view body measurement analysis
- Professional web interface
- Real-time skeleton visualization
- Export results in multiple formats
- Mobile-responsive design

## Support

Check the setup guide for detailed deployment instructions.
"""
    
    create_file('README.md', readme)
    
    # Check if existing measurement files exist
    existing_files = []
    for file in ['main.py', 'body_detector.py', 'measurement_engine.py', 'config.py', 'utils.py']:
        if Path(file).exists():
            existing_files.append(file)
    
    if existing_files:
        print(f"\n📋 Found existing measurement files: {', '.join(existing_files)}")
        print("🔄 These should be moved to the src/ directory for the web system to work.")
        
        move_files = input("\n❓ Move these files to src/ directory automatically? (y/n): ")
        
        if move_files.lower() == 'y':
            for file in existing_files:
                src_path = Path('src') / file
                shutil.move(file, src_path)
                print(f"📦 Moved {file} → {src_path}")
    
    print("\n" + "=" * 60)
    print("🎉 Setup Complete!")
    print("=" * 60)
    
    print("\n📝 Next Steps:")
    print("1. Save the Flask backend code as 'app.py'")
    print("2. Save the web interface code as 'web_interface.html'")
    print("3. Ensure your measurement modules are in the src/ directory")
    print("4. Test locally: python app.py")
    print("5. Deploy to your chosen platform (Railway recommended)")
    
    print(f"\n🌐 Deployment Options:")
    print("• Railway: https://railway.app (recommended)")
    print("• Render: https://render.com")
    print("• Heroku: https://heroku.com")
    
    print(f"\n📁 Project structure created:")
    for directory in directories:
        print(f"  📁 {directory}/")
    print("  📄 requirements.txt")
    print("  📄 Procfile")
    print("  📄 runtime.txt")
    print("  📄 .gitignore")
    print("  📄 README.md")
    
    # Check if git is available
    try:
        subprocess.run(['git', '--version'], capture_output=True, check=True)
        init_git = input("\n❓ Initialize git repository? (y/n): ")
        if init_git.lower() == 'y':
            subprocess.run(['git', 'init'], check=True)
            print("✅ Git repository initialized")
            print("💡 Remember to add your files and commit before deploying!")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("\n⚠️ Git not found. You'll need to initialize git manually for deployment.")

def check_dependencies():
    """Check if required dependencies are available"""
    print("🔍 Checking system dependencies...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major == 3 and python_version.minor >= 8:
        print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    else:
        print(f"⚠️ Python {python_version.major}.{python_version.minor} (recommend 3.8+)")
    
    # Check pip
    try:
        import pip
        print("✅ pip available")
    except ImportError:
        print("❌ pip not found")
    
    # Check if in virtual environment (recommended)
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Virtual environment detected")
    else:
        print("💡 Consider using a virtual environment")
        create_venv = input("❓ Create virtual environment now? (y/n): ")
        if create_venv.lower() == 'y':
            try:
                subprocess.run([sys.executable, '-m', 'venv', 'venv'], check=True)
                print("✅ Virtual environment created as 'venv'")
                print("💡 Activate it with:")
                if os.name == 'nt':  # Windows
                    print("   venv\\Scripts\\activate")
                else:  # Mac/Linux
                    print("   source venv/bin/activate")
            except subprocess.CalledProcessError:
                print("❌ Failed to create virtual environment")

def main():
    """Main setup function"""
    print("🎯 AI Body Measurement System - Web Setup")
    print("=" * 60)
    
    # Check if we're in the right place
    current_dir = Path.cwd()
    print(f"📍 Current directory: {current_dir}")
    
    proceed = input("\n❓ Set up web system in this directory? (y/n): ")
    if proceed.lower() != 'y':
        print("👋 Setup cancelled.")
        return
    
    # Check dependencies
    check_dependencies()
    
    print("\n" + "=" * 60)
    
    # Set up project
    setup_project()
    
    print("\n🎊 Ready for deployment!")
    print("📖 Check the setup guide for detailed deployment instructions.")

if __name__ == "__main__":
    main()