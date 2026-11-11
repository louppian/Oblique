# Oblique

서버 작업 경로: `/shared/home/mai/Taehun/Uncertainty/Oblique`  
Windows 작업 경로: `D:\Oblique`  
공유 저장소: https://github.com/louppian/Oblique

## 서버 최초 연결

Git과 GitHub 읽기 권한이 있는 서버에서 실행합니다.

```bash
curl -fL https://raw.githubusercontent.com/louppian/Oblique/main/setup_server.sh -o /tmp/oblique_setup_server.sh
bash /tmp/oblique_setup_server.sh
```

스크립트는 서버 작업 폴더에서 origin/main을 연결합니다. 기존 저장소는 main에서 fast-forward로 갱신합니다. 기존 파일과 원격 파일이 충돌하면 Git이 중단하므로 해당 파일을 검토한 뒤 다시 실행합니다. 다른 origin이나 다른 현재 브랜치를 발견하면 경로와 브랜치를 확인하도록 중단합니다.

## 서버와 Windows 사이 코드 전달

한쪽에서 수정한 파일을 명시적으로 선택해 커밋하고 push한 뒤 다른 쪽에서 pull합니다. 먼저 현재 변경 사항을 확인합니다.

```bash
git status --short
git add <수정한_코드_경로>
git commit -m "Describe the change"
git push origin main
```

반대쪽 작업 폴더에서:

```bash
git pull --ff-only
```

서버는 `cd /shared/home/mai/Taehun/Uncertainty/Oblique`, Windows PowerShell은 `Set-Location D:\Oblique`로 작업 폴더에 진입합니다. Push에는 각 환경의 GitHub 인증과 Git 사용자 설정이 필요합니다.

## 로컬 데이터

`cache/`, `ckpt/`, `data/`는 모든 깊이에서 Git 제외 대상입니다. 각 머신에서 별도로 준비합니다. Git으로 전달되는 대상은 명시적으로 커밋한 파일입니다. 캐시·가중치·데이터의 파일 동기화는 별도 전송으로 수행합니다.

연구 코드는 `code/`, 외부 소스는 `third_party/`에 둡니다. 이 초기 연결 커밋에는 저장소 설정과 연결 스크립트가 포함됩니다.
