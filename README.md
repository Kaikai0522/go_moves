# 使用方法
## 下載程式碼及安裝環境
輸入以下指令來下載程式碼
```cpp=
git clone https://github.com/Kaikai0522/go_moves.git
```

輸入以下指令來安裝環境，env_name改為自訂環境名稱
```cpp=
conda create --name {env_name} python=3.9
conda activte {env_name}
cd go_moves
pip install -r requirements.txt
```

## 下載Sabaki
https://sabaki.yichuanshen.de/

## 設定Sabaki

### 1. 打開左側Engines Sidebar
Engines → Show Engines Sidebar<br>
![show_engines_sidebar](./images/show_engines_sidebar.png)

### 2. 打開右側Game Tree
View → Show Game Tree<br>
![show_game_tree](./images/show_game_tree.png)

### 3. 設定圍棋引擎
點擊Attach Engine按鈕(播放按鈕) → Manages Engines → Add → 設置引擎 → Close (Close後會自動儲存)

**如使用虛擬環境安裝套件，則python路徑為虛擬環境python路徑 ex: "C:\Users\88697\anaconda3\envs\\{env_name}\python.exe"**
![設定引擎](./images/設定引擎.png)

### 4. 選擇圍棋引擎
點擊Attach Engine按鈕(播放按鈕) → 選擇go moves引擎<br>
![選擇引擎](./images/選擇引擎.png)

### 5. 設定對手棋子顏色
對引擎名稱點擊右鍵 → 選擇Set as Black/White Player<br>
![設定對手顏色](./images/設定對手顏色.png)

做完以上設定就可以開始跟圍棋引擎對弈

### 6. 儲存棋譜
點擊右下角視窗的Save File按鈕(如下圖)，即可在sgf資料夾中看到存好的棋譜(棋譜命名為目前時間)。<br>
![著手顯示介面](./images/著手顯示介面.png)



