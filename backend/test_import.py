import traceback
try:
    from app.main import app
except Exception as e:
    open('err.log', 'w').write(traceback.format_exc())
