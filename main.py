from fastapi import FastAPI, Depends, HTTPException, status, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker, Session
from pydantic import BaseModel
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
import random
from slowapi import Limiter
from slowapi.util import get_remote_address
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


DATABASE_URL = "sqlite:///./test.db"

Base = declarative_base()
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)



SECRET_KEY = "09bbf16e0c6d4538b36a35d9b8d38b07b8df1d9f0c6d9537b5cb30f1d7e9e17d"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

fake_users_db = {
    "NikhilN": {
        "username": "NikhilN",
        "full_name": "Nikhil Nagalla",
        "email": "nikhilnagalla08@gmail.com",
        "hashed_password": pwd_context.hash("Nikhil123"),
        "disabled": False,
    }
}

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: str | None = None

class User(BaseModel):
    username: str
    email: str | None = None
    full_name: str | None = None
    disabled: bool | None = None

class UserInDB(User):
    hashed_password: str


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)

def authenticate_user(fake_users_db, username: str, password: str):
    user = get_user(fake_users_db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = get_user(fake_users_db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

class Robot(Base):
    __tablename__ = "robots"
    id = Column(Integer, primary_key=True, index=True)
    x = Column(Integer)
    y = Column(Integer)
    orientation = Column(String)
    battery_level = Column(Integer)
    status = Column(String)
    tasks = relationship("Task", back_populates="robot")

class Task(Base):
    __tablename__ = "tasks"
    id = Column(Integer, primary_key=True, index=True)
    type = Column(String)
    status = Column(String)
    robot_id = Column(Integer, ForeignKey('robots.id'))
    robot = relationship("Robot", back_populates="tasks")

Base.metadata.create_all(bind=engine)

class RobotCreate(BaseModel):
    x: int
    y: int
    orientation: str
    battery_level: int

class TaskCreate(BaseModel):
    type: str

from fastapi import HTTPException, status

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    print(f"Received form data: {form_data}")
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user.username}, expires_delta=access_token_expires
        )
        print(f"Generated access token: {access_token}")
        return {"access_token": access_token, "token_type": "bearer"}
    except Exception as e:
        print(f"Error occurred: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create access token",
        ) from e
    

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.post("/robots/")
def create_robot(robot: RobotCreate, db: Session = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    db_robot = Robot(x=robot.x, y=robot.y, orientation=robot.orientation, battery_level=robot.battery_level, status="idle")
    db.add(db_robot)
    db.commit()
    db.refresh(db_robot)
    return db_robot

@app.post("/tasks/")
def create_task(task: TaskCreate, db: Session = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    db_task = Task(type=task.type, status="pending")
    db.add(db_task)
    db.commit()
    db.refresh(db_task)
    return db_task

@app.post("/assign-task/")
def assign_task(robot_id: int, task_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    robot = db.query(Robot).filter(Robot.id == robot_id).first()
    task = db.query(Task).filter(Task.id == task_id).first()
    if not robot or not task:
        raise HTTPException(status_code=404, detail="Robot or Task not found")
    task.robot_id = robot.id
    task.status = "in-progress"
    robot.status = "busy"
    db.commit()
    return {"message": "Task assigned"}

@app.get("/robots/")
def read_robots(skip: int = 0, limit: int = 10, db: Session = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    robots = db.query(Robot).offset(skip).limit(limit).all()
    return robots

@app.get("/tasks/")
def read_tasks(skip: int = 0, limit: int = 10, db: Session = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    tasks = db.query(Task).offset(skip).limit(limit).all()
    return tasks

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await manager.broadcast(f"Message text was: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast("Client left the chat")

def perform_task(task_id: int, db: Session):
    task = db.query(Task).filter(Task.id == task_id).first()
    # Simulate task execution
    for i in range(100):
        task.progress += 1
        db.commit()
    task.status = "completed"
    db.commit()

@app.post("/perform-task/")
def perform_task_endpoint(task_id: int, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    background_tasks.add_task(perform_task, task_id, db)
    return {"message": "Task started"}

limiter = Limiter(key_func=get_remote_address)

@app.on_event("startup")
async def startup():
    app.state.limiter = limiter

@app.middleware("http")
async def add_process_time_header(request, call_next):
    response = await call_next(request)
    return response

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/metrics")
def metrics():
    # Implement metrics collection and return here
    pass



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
