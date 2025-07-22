import subprocess
import sys
from datetime import datetime

def run(script):
    print(f"{datetime.now():%H:%M:%S} → running {script}")
    subprocess.run([sys.executable, script], check=True)

if __name__ == "__main__":
    try:
        # 1) Сначала собираем новую историю по биржам
        run("generate_history.py")

        # 2) Размечаем сделки из лога (на основе свежего bot_log.log)
        run("label_trades.py")
        
        # НОВОЕ: 2.5) Генерируем идеальные метки из истории
        run("generate_ideal_labels.py") # <-- ДОБАВЬТЕ ЭТУ СТРОКУ

        # 3) Превращаем историю + разметку в обучающий датасет
        run("feature_engineering.py")

        # 4) Обучаем и сохраняем модель
        run("train_model.py")

        print(f"{datetime.now():%H:%M:%S} ALL DONE")
    except subprocess.CalledProcessError as e:
        print(f"ERROR in {e.cmd}: exit {e.returncode}")
        sys.exit(1)