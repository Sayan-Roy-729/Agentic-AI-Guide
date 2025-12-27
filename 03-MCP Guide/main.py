import os
import aiosqlite # Change: sqlite3 -> aiosqlite for async supports
import tempfile
from fastmcp import FastMCP

# Define database and categories file paths
TEMP_DIR = tempfile.gettempdir()
DB_PATH = os.path.join(TEMP_DIR, "expenses.db")
CATEGORIES_PATH = os.path.join(os.path.dirname(__file__), "categories.json")

# Create a FastMCP server instance
mcp = FastMCP(name = "Expense_Tracker")


async def init_db():
    try:
        async with aiosqlite.connect(DB_PATH) as conn:
            await conn.execute("PRAGMA journal_mode=WAL;")
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS expenses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    amount REAL NOT NULL,
                    category TEXT NOT NULL,
                    subcategory TEXT DEFAULT '',
                    note TEXT DEFAULT ''
                )
            """)
            # Test write access

            await conn.execute("INSERT OR IGNORE INTO expenses (date, amount, category) VALUES (DATE('now'), 0, 'Test')")
            await conn.execute("DELETE FROM expenses WHERE category = 'Test'")
            print("Database initialized successfully with write access.")
    except Exception as e:
        print(f"Error initializing database: {e}")

init_db()

@mcp.tool
async def add_expense(date, amount, category, subcategory="", note=""):
    """
    Add a new expense to the database.
    Args:
        date (str): The date of the expense in the format 'YYYY-MM-DD'.
        amount (float): The amount of the expense.
        category (str): The category of the expense.
        subcategory (str, optional): The subcategory of the expense. Defaults to an empty string.
        note (str, optional): A note or description for the expense. Defaults to an empty string.
    Returns:
        dict: A dictionary containing the status and the ID of the newly added expense.
    """
    try:
        async with aiosqlite.connect(DB_PATH) as conn:
            cur = await conn.execute("""
                INSERT INTO expenses (date, amount, category, subcategory, note)
                VALUES (?, ?, ?, ?, ?)
            """, (date, amount, category, subcategory, note))
            expense_id = cur.lastrowid
            await conn.commit()
            return {"status": "success", "id": expense_id, "message": "Expense added successfully."}
    except aiosqlite.OperationalError as e:
        if "readonly" in str(e).lower():
            return {"status": "error", "message": "Database is in read-only mode. Check file permissions."}
        return {"status": "error", "message": f"Database error: {str(e)}"}
    except Exception as e:
        return {"status": "error", "message": f"An unexpected error occurred: {str(e)}"}

@mcp.tool
async def list_expenses(start_date: str = None, end_date: str = None) -> float:
    """
    List all expenses in the database.
    Args:
        start_date (str, optional): The start date for filtering expenses in the format 'YYYY-MM-DD'. Defaults to None.
        end_date (str, optional): The end date for filtering expenses in the format 'YYYY-MM-DD'. Defaults to None.
    Returns:
        list: A list of dictionaries, each representing an expense record.
    """
    try:
        async with aiosqlite.connect(DB_PATH) as conn:
            cur = await conn.execute("""
                    SELECT id, date, amount, category, subcategory, note 
                    FROM expenses
                    WHERE date BETWEEN ? AND ?
                    ORDER BY date DESC, id DESC
                """, (start_date, end_date))
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, r)) for r in await cur.fetchall()]
    except Exception as e:
        return {"status": "error", "message": f"Error listing expenses: {str(e)}"}

@mcp.tool
async def summarize(start_date: str, end_date: str, category: str = None):
    """
    Summarize expenses by category within a date range.
    Args:
        start_date (str): The start date for filtering expenses in the format 'YYYY-MM-DD'.
        end_date (str): The end date for filtering expenses in the format 'YYYY-MM-DD'.
        category (str, optional): The category to filter expenses. Defaults to None.
    Returns:
        list: A list of dictionaries, each representing a summarized expense record.
    """
    try:
        async with aiosqlite.connect(DB_PATH) as conn:
            query = """
                SELECT category, SUM(amount) as total_amount, COUNT(*) as count
                FROM expenses
                WHERE date BETWEEN ? AND ?
            """
            params = [start_date, end_date]
            if category:
                query += " AND category = ?"
                params.append(category)

            query += " GROUP BY category ORDER BY total_amount DESC"

            cur = await conn.execute(query, params)
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, r)) for r in await cur.fetchall()]
    except Exception as e:
        return {"status": "error", "message": f"Error summarizing expenses: {str(e)}"}

@mcp.resource("expense:///categories", mime_type="application/json")
def categories():
    # read fresh each time so you can edit the file without restarting the server
    with open(CATEGORIES_PATH, "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    mcp.run()
