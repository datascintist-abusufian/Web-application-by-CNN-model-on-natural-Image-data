# Create .streamlit directory
mkdir .streamlit

# Create config.toml with content
echo '[theme]
primaryColor = "#F63366"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[server]
maxUploadSize = 200

[browser]
gatherUsageStats = false' > .streamlit/config.toml

# Update requirements.txt
echo 'streamlit>=1.28.0
tensorflow>=2.15.0
Pillow>=10.0.0
numpy>=1.24.3
plotly>=5.17.0
requests>=2.31.0
matplotlib>=3.7.1
scikit-learn>=1.3.0
protobuf>=3.20.0' > requirements.txt

# Add new files to git
git add .streamlit/
git add requirements.txt

# Commit changes
git commit -m "Add Streamlit configuration and update requirements"

# Push to GitHub
git push origin main
