import pandas as pd

MAX_INDEX = 1400

def read_csv_file(file_path):
    """
    csv 파일 읽기용 함수
    """
    data = pd.read_csv(file_path)
    return data

def inspection():
    '''
    우선, 검사를 시작할 index를 입력하세요.
    
    검사할 텍스트가 주어지면, 텍스트를 읽고 적합한지 판단하세요.
    수정할 필요가 없다면 'p'를 입력하세요.
    수정이 필요하다면, 수정된 텍스트를 입력하세요.
    이전 index로 돌아가고 싶다면 'b'를 입력하세요.
    
    이후 주제 후보들의 목록이 주어집니다.
    적절한 주제에 해당하는 숫자를 입력하세요.
    0부터 6까지의 숫자로 입력하세요.
    
    inspected_train.csv에는 내가 검수한 뒤 변경한 내용들이 저장됩니다.
    noise_train.csv에는 내가 변경한 noise 항목들만 저장됩니다. 단, 변경하기 전 원래 데이터로 저장됩니다.
    자신이 검사할 분량이 끝나면 자동으로 종료됩니다.
    '''
    train_path = "../data/train.csv"
    noise_train = "noise_train.csv"
    inspected_train = "inspected_train.csv"
    
    try:
        existing_data = pd.read_csv(inspected_train, dtype=str)
    except FileNotFoundError:
        existing_data = pd.DataFrame()
        
    try:
        noise_data = pd.read_csv(noise_train, dtype=str)
    except FileNotFoundError:
        noise_data = pd.DataFrame()
    
    existing_data = existing_data.reindex(columns=['ID', 'text', 'target', 'url', 'date'], index=range(7000), fill_value='')
    noise_data = existing_data.reindex(columns=['ID', 'text', 'target', 'url', 'date'], index=range(7000), fill_value='')
    data = read_csv_file(train_path)
    topic = {0: "정치", 1: "경제", 2: "사회", 3: "생활문화", 4: "세계" , 5:"IT과학" , 6: "스포츠"}
    
    while True:
        index = int(input("검사를 시작할 Index를 입력하세요. \n시작 Index: "))

        # 올바른 index 검사
        if 0 <= index <= 7000:
            break 
        else:
            print("올바르지 않은 입력입니다. 0에서 7000 사이의 숫자를 입력하세요.")

    start = index
    
    while (index % MAX_INDEX) != 0 or start == index:
    
        ID_now = data.iloc[index]['ID']
        text_now = data.iloc[index]['text']
        target_now_num = data.iloc[index]['target']
        target_now = topic[int(target_now_num)]
        last = 1400 - (index % MAX_INDEX)
        
        print(f"남은 갯수: {last}")
        print("다음 텍스트의 내용은 정확한가요? 수정이 필요하다면 수정된 텍스트를 입력해 주세요. ('p' : 패스, 'b' : 한칸 전으로)\n")
        text = input(f"{text_now}\n")
        
        if text == 'p':
            text = text_now
        elif text == 'b':
            index -= 1
            continue
        else:
            noise_data.loc[index, :] = data.loc[index, :]
        
        print("----------------------------------------------------------------------------------------------------------------")
        print("----------------------------------------------------------------------------------------------------------------\n")
        
        valid_target = False
        while not valid_target:
            print(f"알맞은 주제를 라벨링해 주세요. 기존에 라벨링된 주제는 '{target_now}({target_now_num})'입니다.\n")
            print(f"{text}\n")
            target = input("정치(0)  경제(1)  사회(2)  생활문화(3)  세계(4)  IT과학(5)  스포츠(6)\n")
            
            if target == "p":
                target = target_now_num
            
            try:
                target_int = int(target)
                if 0 <= target_int <= 6:
                    valid_target = True
                else:
                    print("\n입력된 라벨이 범위를 벗어났습니다. 0에서 6 사이의 값을 입력하세요.")
            except ValueError:
                print("\n입력된 값이 유효한 정수가 아닙니다. 0에서 6 사이의 값을 입력하세요.")
        
        print("----------------------------------------------------------------------------------------------------------------")
        print("----------------------------------------------------------------------------------------------------------------\n")
        
        print(f"다음과 같이 라벨링되었습니다:\n{text} - 주제 : {topic[target_int]}\n")
        data.loc[index, 'text'] = text
        data.loc[index, 'target'] = str(target_int)
        
        print("----------------------------------------------------------------------------------------------------------------")
        print("----------------------------------------------------------------------------------------------------------------\n")
        
        existing_data.loc[index, :] = data.loc[index, :]
        existing_data.to_csv('inspected_train.csv', index=False)
        noise_data.to_csv('noise_train.csv', index=False)
        
        index += 1
    
    print("끝!!!")
    


if __name__ == "__main__":
    inspection()