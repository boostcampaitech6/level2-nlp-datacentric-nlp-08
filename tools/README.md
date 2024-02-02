# 1. apt 업데이트
```
apt update
```

# 2. Chrome 설치

## 2-1. Chrome debian 파일 다운로드 및 설치
```
wget https://dl.google.com/linux/chrome/deb/pool/main/g/google-chrome-stable/google-chrome-stable_121.0.6167.85-1_amd64.deb
apt install ./google-chrome-stable_121.0.6167.85-1_amd64.deb
```

## 2-2. Chrome 버전 확인
```
google-chrome --version
```
121.0.6167.85 확인
```
rm ./google-chrome-stable_121.0.6167.85-1_amd64.deb
```

# 3. Chrome Driver 설치
```
wget https://edgedl.me.gvt1.com/edgedl/chrome/chrome-for-testing/121.0.6167.85/linux64/chromedriver-linux64.zip
unzip ./chromedriver-linux64.zip
chmod +x ./chromedriver-linux64/chromedriver
rm ./chromedriver-linux64.zip
```

# 4. 필요한 패키지 설치
```
pip install selenium
pip install webdriver_manager
```