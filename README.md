1. Di chuyển đến đường dẫn có thư mục:
```bash
cd avt_preprocess
```
2. Sửa nội dung file config
- Sửa các dòng thông tin về máy chủ database, máy chủ ftp
```bash
{
    "database": {
        "host": "118.70.x.x",
        "database": "avt",
        "user": "postgres",
        "password": "xxxx",
        "port": 18944
    },
    "ftp": {
        "host": "118.70.x.x",
        "port": 18921,
        "user": "avt",
        "password": "x"
    },
    "modules": {
        "1":  "modules01_correctionmain",
        "2":  "modules02_pre_processmain",
        "3":  "modules03_cloud_removemain",
        "4":  "modules04_enhancementmain",
        "5": "modules05_detectionmain",
        "6":  "modules06_classificationmain",
        "7":  "modules07_object_findermain",
        "8":  "python testnon_responding.py"
    }
}
```
3. Chạy lệnh build docker
```bash
docker compose build
```
4. Chạy lệnh run docker
```bash
docker compose run -d
```
