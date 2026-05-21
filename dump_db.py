import sqlite3

conn = sqlite3.connect("data/insurance.db")
conn.row_factory = sqlite3.Row

tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
print("=" * 60)
print("DATABASE TABLES")
print("=" * 60)
for t in tables:
    name = t[0]
    count = conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
    print(f"\n[{name}]  {count} rows")
    if count > 0:
        rows = conn.execute(f"SELECT * FROM {name} LIMIT 5").fetchall()
        for r in rows:
            print(" ", dict(r))

conn.close()
