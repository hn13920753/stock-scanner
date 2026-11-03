"""
模擬交易的存檔：把交易紀錄存成 GitHub 倉庫裡的一個 JSON 檔（獨立分支 paper-data，不會觸發 App 重新部署）。

Streamlit Cloud 重新啟動會清掉記憶體，所以必須存到外部。需要在 App 的 Secrets 設定：
    GITHUB_TOKEN = "github_pat_..."    # 只開 stock-scanner 倉庫 Contents 讀寫權限
可選：
    PAPER_REPO = "hn13920753/stock-scanner"
    PAPER_BRANCH = "paper-data"
沒有金鑰時只能讀（公開倉庫），寫入會失敗並提示設定。
"""
import base64
import json

import requests

API = "https://api.github.com"
DEFAULT_REPO = "hn13920753/stock-scanner"
DEFAULT_BRANCH = "paper-data"
PATH = "paper/trades.json"


class Store:
    def __init__(self, token: str | None = None, repo: str = DEFAULT_REPO, branch: str = DEFAULT_BRANCH):
        self.token, self.repo, self.branch = token, repo, branch
        self.s = requests.Session()
        self.s.headers.update({"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"})
        if token:
            self.s.headers["Authorization"] = f"Bearer {token}"

    @property
    def can_write(self) -> bool:
        return bool(self.token)

    def load(self):
        """回傳 (資料 dict, sha)。檔案不存在時回傳預設值、sha=None。"""
        r = self.s.get(f"{API}/repos/{self.repo}/contents/{PATH}", params={"ref": self.branch}, timeout=10)
        if r.status_code == 404:
            return {"capital": 1000000, "disc": 1.0, "trades": []}, None
        r.raise_for_status()
        j = r.json()
        data = json.loads(base64.b64decode(j["content"]).decode("utf-8"))
        data.setdefault("capital", 1000000)
        data.setdefault("disc", 1.0)
        data.setdefault("trades", [])
        return data, j["sha"]

    def save(self, data: dict, sha: str | None, message: str):
        """寫回。別處先改過（sha 不符）會丟出 ConflictError。回傳新的 sha。"""
        if not self.token:
            raise PermissionError("還沒設定存檔金鑰（GITHUB_TOKEN）")
        body = {"message": message, "branch": self.branch,
                "content": base64.b64encode(json.dumps(data, ensure_ascii=False, indent=1).encode("utf-8")).decode()}
        if sha:
            body["sha"] = sha
        r = self.s.put(f"{API}/repos/{self.repo}/contents/{PATH}", json=body, timeout=15)
        if r.status_code in (409, 422) and "sha" in r.text:
            raise ConflictError("紀錄在別的畫面被改過，請重新整理再試")
        if r.status_code == 401:
            raise PermissionError("存檔金鑰無效或已過期，請重新建立")
        if r.status_code == 403:
            raise PermissionError("存檔金鑰沒有寫入權限（要開 Contents：Read and write）")
        if r.status_code == 404:
            raise PermissionError("存檔金鑰沒有開放 stock-scanner 倉庫，或分支 paper-data 不存在")
        r.raise_for_status()
        return r.json()["content"]["sha"]


class ConflictError(RuntimeError):
    pass
