# 프로그래머스 데브코스 final project : 수리 (Su Re)


## Introduction
- 과거에 명작 영화들이 초고화질(super-resolution)로 리마스터링 되어 재개봉되고 있음
- 과거에 찍은 사진들은 촬영 관련 hardware 기술 부재로 저화질인 반면, 현재는 hardware 기술이 상당히 좋아져 사진의 품질이 대폭 상향 평준화
- 이에 따라 개개인이 가지고 있는, 다시 찍을 수 없는 저화질의 사진들에 대해 고화질로 만들고 싶은 needs가 있을 거라 판단
- 추억이 담긴 사진들이 대부분 인물에 국한되어 있는데, 인물 뿐만 아니라 
동물, 사물, 건물, 배경 모든 분야에 사진에 대해서도 고화질로 만들 수 있는 플랫폼 제공

## Requirements
OS : AWS Linux 2  
Docker : v20.10.7  
Docker-compose : v2.4.1  
Nvidia Driver : 470.103.01    
CUDA : v11.4



## Build & Run
package.json파일의 proxy주소를 docker-compose로 만든 컨테이너의 network 의 gateway로 변경  
### 주소 찾는 법 및 proxy 주소 설정  


`docker network ls` 컨테이너가 연결된 네트워크 ID 확인 (super resolution)  


`docker network inspect [Network ID]` 네트위크의 주소 확인  
```
"IPAM": {  
  "Driver": "default",  
    "Options": null,  
      "Config": [  
        {  
          "Subnet": "???.??.???.?",  
          "Gateway": "???.??.?.?"  <= 이부분을 check
        }  
     ]  
 },  
 ```
 
 `package.json` 파일의 proxy 주소를 위의 Gateway로 변경
 

`npm run build`  
패키지 파일 생성


`docker-compose --build -d`  
Docker 이미지 생성, 컨테이너 생성 + 빌드 + 실행  




## Example
Main Page
![Example](./Images/example.png)

Crop
![Crop](./Images/Crop.png)

Get SR_Images

![Load](./Images/Load.png)


## Result
![Compare](./Images/compare.png)


## Reference
Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data: https://arxiv.org/pdf/2107.10833.pdf


