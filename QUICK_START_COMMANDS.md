# 🚀 Quick Start Commands - Moaz Mohamed

## GitHub Repository Setup

### 1. Initialize Git Repository
```bash
python init_git_repo.py
```

### 2. Create GitHub Repository
- Go to: https://github.com/moazmo
- Click "New repository"
- Repository name: `elevvo-pathways-ml-internship`
- Description: `🚀 Professional ML internship portfolio at Elevvo Pathways - 5 comprehensive projects spanning classical ML to deep learning with production deployments`
- Set to **Public**
- **Don't** initialize with README (we have our own)

### 3. Push to GitHub
```bash
# Check existing remotes
git remote -v

# Option A: Update existing remote (recommended)
git remote set-url origin https://github.com/moazmo/elevvo-pathways-ml-internship.git

# Option B: Remove and re-add remote (if Option A doesn't work)
# git remote remove origin
# git remote add origin https://github.com/moazmo/elevvo-pathways-ml-internship.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### 4. Verify Upload
- Check that all files uploaded correctly
- Verify Git LFS files are properly tracked
- Ensure README displays correctly

## Repository URL
**Final Repository**: https://github.com/moazmo/elevvo-pathways-ml-internship

## Quick Clone Commands for Others
```bash
# Clone with Git LFS support
git lfs clone https://github.com/moazmo/elevvo-pathways-ml-internship.git
cd elevvo-pathways-ml-internship

# Or regular clone then pull LFS
git clone https://github.com/moazmo/elevvo-pathways-ml-internship.git
cd elevvo-pathways-ml-internship
git lfs pull
```

## Individual Project Setup
```bash
# Task 1: Customer Segmentation
cd Task_1
pip install -r requirements.txt
jupyter notebook customer_segmentation_analysis.ipynb

# Task 2: Forest Cover Classification
cd Task_2
pip install -r requirements.txt
jupyter notebook forest_cover_classification.ipynb

# Task 3: Loan Approval Prediction
cd Task_3
pip install -r requirements.txt
python main_pipeline.py

# Task 4: Walmart Sales Forecasting
cd Task_4
pip install -r requirements.txt
python start_full_system.py --frontend

# Task 5: Traffic Sign Recognition
cd Task_5
pip install -r requirements.txt
python test_app.py
cd webapp && python app.py
```

## Final Checklist
- [x] GitHub username updated (moazmo)
- [x] Repository URLs corrected
- [x] License updated with your name
- [x] Update email address in contact sections
- [x] Update LinkedIn profile URL
- [x] Add portfolio website URL
- [x] Add internship duration dates (August 2025)
- [x] Task_4 and Task_5 successfully published
- [x] All placeholder text removed

## Next Steps After Publishing
1. Update LinkedIn profile with repository link
2. Add to resume/CV
3. Share with professional network
4. Consider creating a blog post about the projects
5. Keep repository updated and maintained

---

**Ready to showcase your professional ML expertise! 🚀**