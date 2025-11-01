[app]

# 基本資訊
title = 雞湯精靈
package.name = chickensoup
package.domain = org.baiyanchen
version = 0.1
orientation = portrait

# 主程式目錄
source.dir = .

# 包含檔案副檔名
source.include_exts = py,csv,pt,png,jpg,kv,atlas

# 排除特定資料夾或檔案
source.exclude_dirs = tests,venv
source.exclude_exts = 
source.exclude_patterns = 

# Python 模組依賴
requirements = python3,kivy,pandas,torch,sentence-transformers

# 圖片與 icon（可選）
#icon.filename = %(source.dir)s/assets/icon.png
#presplash.filename = %(source.dir)s/assets/presplash.png

# 是否全螢幕
fullscreen = 0

# -----------------------------------------------------------------------------
# Android 特定設定
# -----------------------------------------------------------------------------
android.archs = arm64-v8a, armeabi-v7a
android.minapi = 21
android.debug_artifact = apk
android.release_artifact = aab

# 離線 App，不需要網路權限
#android.permissions =

# -----------------------------------------------------------------------------
# iOS 特定設定
# -----------------------------------------------------------------------------
ios.kivy_ios_url = https://github.com/kivy/kivy-ios
ios.kivy_ios_branch = master
ios.ios_deploy_url = https://github.com/phonegap/ios-deploy
ios.ios_deploy_branch = 1.10.0
ios.codesign.allowed = false

# -----------------------------------------------------------------------------
# Buildozer 全域設定
# -----------------------------------------------------------------------------
[buildozer]
log_level = 2
warn_on_root = 1

# Android SDK 路徑
android.sdk_path = /home/baiyanchen/Android/Sdk

# Android NDK 路徑
android.ndk_path = /home/baiyanchen/Android/Sdk/ndk/27.0.12077973

# Android API level
android.api = 33
android.minapi = 21

