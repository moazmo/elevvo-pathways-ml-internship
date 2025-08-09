# 🔧 Troubleshooting Guide - Elevvo Pathways ML Portfolio

## 🚨 Common Issues and Solutions

### 1. "error: remote origin already exists"

**Problem**: When running `git remote add origin <url>`, you get this error.

**Solution**: You already have a remote origin configured. Update it instead:

```bash
# Check current remotes
git remote -v

# Update the existing remote URL
git remote set-url origin https://github.com/moazmo/elevvo-pathways-ml-internship.git

# Verify the change
git remote -v

# Now push normally
git branch -M main
git push -u origin main
```

**Alternative Solution** (if the above doesn't work):
```bash
# Remove existing remote
git remote remove origin

# Add new remote
git remote add origin https://github.com/moazmo/elevvo-pathways-ml-internship.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### 2. Git LFS Files Not Uploading

**Problem**: Large files aren't uploading to GitHub or showing as text pointers.

**Solution**:
```bash
# Ensure Git LFS is installed
git lfs install

# Check LFS tracking
git lfs track

# Pull any existing LFS files
git lfs pull

# Push LFS files
git lfs push origin main
```

### 3. "Repository not found" Error

**Problem**: Git can't find your repository on GitHub.

**Solution**:
1. Make sure you've created the repository on GitHub first
2. Check the repository name is exactly: `elevvo-pathways-ml-internship`
3. Verify your GitHub username is correct: `moazmo`
4. Ensure the repository is public (not private)

### 4. Authentication Issues

**Problem**: Git asks for username/password or shows authentication errors.

**Solution**:
```bash
# Use GitHub CLI (recommended)
gh auth login

# Or configure Git with your credentials
git config --global user.name "Moaz Mohamed"
git config --global user.email "your.email@example.com"

# For HTTPS, you might need a Personal Access Token
# Go to GitHub Settings > Developer settings > Personal access tokens
```

### 5. Large Files Rejected by GitHub

**Problem**: Files larger than 100MB are rejected even with Git LFS.

**Solution**:
```bash
# Check file sizes
find . -type f -size +50M -exec ls -lh {} \;

# Remove large files from Git history if needed
git filter-branch --force --index-filter 'git rm --cached --ignore-unmatch path/to/large/file' --prune-empty --tag-name-filter cat -- --all

# Or use BFG Repo-Cleaner for better performance
# java -jar bfg.jar --strip-blobs-bigger-than 100M
```

### 6. Virtual Environment Issues

**Problem**: Virtual environment files are being tracked by Git.

**Solution**:
```bash
# Remove from Git if already tracked
git rm -r --cached .venv/
git rm -r --cached venv/
git rm -r --cached Task_*/.venv/

# Commit the removal
git commit -m "Remove virtual environment files"

# .gitignore should already handle this for future
```

### 7. Jupyter Notebook Checkpoints

**Problem**: `.ipynb_checkpoints` folders are being tracked.

**Solution**:
```bash
# Remove from Git
find . -name ".ipynb_checkpoints" -exec git rm -r --cached {} \;

# Commit the removal
git commit -m "Remove Jupyter checkpoint files"
```

### 8. Permission Denied (Windows)

**Problem**: Permission denied errors when running scripts.

**Solution**:
```powershell
# Run PowerShell as Administrator
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Or run specific script
powershell -ExecutionPolicy Bypass -File init_git_repo.py
```

### 9. Python Module Not Found

**Problem**: ImportError or ModuleNotFoundError when running projects.

**Solution**:
```bash
# Ensure you're in the correct directory
cd Task_X  # Replace X with task number

# Install requirements
pip install -r requirements.txt

# Check Python path
python -c "import sys; print(sys.path)"

# Add current directory to Python path if needed
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### 10. Port Already in Use

**Problem**: "Port 8000 is already in use" when starting APIs.

**Solution**:
```bash
# Windows - Find and kill process
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux/Mac - Find and kill process
lsof -ti:8000 | xargs kill -9

# Or use different port
uvicorn src.api_server:app --port 8001
```

## 🔍 Diagnostic Commands

### Check Git Status
```bash
git status
git log --oneline -10
git remote -v
git branch -a
```

### Check Git LFS
```bash
git lfs version
git lfs track
git lfs ls-files
git lfs status
```

### Check Python Environment
```bash
python --version
pip list
which python
echo $PYTHONPATH
```

### Check File Sizes
```bash
# Find large files
find . -type f -size +10M -exec ls -lh {} \;

# Check Git LFS files
git lfs ls-files --size
```

## 📞 Getting Help

If you're still having issues:

1. **Check the error message carefully** - it usually tells you exactly what's wrong
2. **Search GitHub Issues** - others might have had the same problem
3. **Check Git documentation** - `git help <command>`
4. **Verify prerequisites** - Python version, Git version, etc.
5. **Try a fresh clone** - sometimes starting over is faster

## 🚀 Success Verification

After fixing issues, verify everything works:

```bash
# 1. Check repository is properly connected
git remote -v

# 2. Check files are properly tracked
git status

# 3. Check LFS files
git lfs ls-files

# 4. Test a small change
echo "# Test" >> test.md
git add test.md
git commit -m "Test commit"
git push origin main
git rm test.md
git commit -m "Remove test file"
git push origin main
```

---

**Most issues are simple configuration problems that can be fixed quickly! 🛠️**