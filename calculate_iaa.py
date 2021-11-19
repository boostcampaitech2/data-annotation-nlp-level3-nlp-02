import pandas as pd
import numpy as np
from fleiss import fleissKappa
import gspread
from oauth2client.service_account import ServiceAccountCredentials


def get_data_from_google_sheet():
    scope = ['https://spreadsheets.google.com/feeds',
            'https://www.googleapis.com/auth/drive']

    #json key file 위치
    json_file_name = './stable-house-327308-c4a6a7447abd.json'

    # json key file을 이용하여 접속
    credentials = ServiceAccountCredentials.from_json_keyfile_name(json_file_name, scope)
    gc = gspread.authorize(credentials)

    #구글 스프레드 시트 주소
    spreadsheet_url = "https://docs.google.com/spreadsheets/d/1Zfhy3xh-2or5dUsSRm9d3yIN4Jhaleh32Rxqejy8hbI/edit#gid=0"

    # 스프레드시트 문서 가져오기
    doc = gc.open_by_url(spreadsheet_url)
    ## gc.create(spreadsheet_name) # 스프레드시트 생성

    values = doc.worksheet("final").get_all_values()
    header, rows = values[0], values[1:]
    return pd.DataFrame(data = rows, columns = header)

def main():
    relation = {"no_relation":1,"org:members":2,"org:founded":3,"related":4,
                "product":5,"org:event":6,"org:field":7,"alternate":8,"location":9,"per:member_of":10,
                "per:title":11,"poh:type":12,"poh:start_date":13,"poh:end_date":14}
    num_classes = len(relation)

    #구글 시트로부터 데이터 불러오기
    data = get_data_from_google_sheet()
    
    #라벨을 숫자로 변환
    result = []
    for row_labels in data['work1'].map(lambda x : [[label]*int(num) for label, num in eval(x).items()]):
        row_labels = sum(row_labels,[])
        row_labels = [relation[label] for label in row_labels]
        result += [row_labels]
    
    
    transformed_result = []
    for i in range(len(result)):
        temp = np.zeros(num_classes)
        for j in range(len(result[i])):
            temp[int(result[i][j]-1)] += 1
        transformed_result.append(temp.astype(int).tolist())

    kappa = fleissKappa(transformed_result,len(result[0]))

if __name__=="__main__":
    main()
