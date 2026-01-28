from core.database import StockDatabase
from config import settings

db = StockDatabase(str(settings.path.DB_PATH))

# 检查表是否存在
with db.connect() as conn:
    tables = conn.execute("SHOW TABLES").fetchall()
    print("数据库中的表:", tables)