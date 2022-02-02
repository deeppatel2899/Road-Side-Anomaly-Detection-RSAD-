from RSAD import db
from RSAD.com.vo.CrossroadVO import CrossroadVO
import datetime

class VideoVO(db.Model):
    __tablename__ = 'videomaster'
    videoId = db.Column('videoId', db.Integer, primary_key=True, autoincrement=True)
    videoPath = db.Column('videoPath',db.String(100))
    videoName = db.Column('videoName',db.String(100), unique=True)
    videoDate = db.Column('videoDate', db.String(100), default=datetime.datetime.now().strftime("%d-%m-%Y"))
    videoTime = db.Column('videoTime', db.String(100), default=datetime.datetime.now().strftime("%H:%M:%S"))
    video_crossroadId = db.Column('video_crossroadId', db.Integer, db.ForeignKey(CrossroadVO.crossroadId))



    def as_dict(self):
        return {
            'videoId': self.videoId,
            'videoName': self.videoName,
            'videoDate': self.videoDate,
            'videoTime': self.videoTime,
            'video_crossroadId':self.video_crossroadId
        }


db.create_all()
