from RSAD import db
from RSAD.com.vo.AreaVO import AreaVO


class CrossroadVO(db.Model):
    __tablename__ = 'crossroadmaster'
    crossroadId = db.Column('crossroadId', db.Integer, primary_key=True, autoincrement=True)
    crossroadName = db.Column('crossroadName', db.String(100))
    crossroad_areaId = db.Column('crossroad_areaId', db.Integer, db.ForeignKey(AreaVO.areaId))



    def as_dict(self):
        return {
            'crossroadId': self.crossroadId,
            'crossroadName': self.crossroadName,
            'crossroad_areaId':self.crossroad_areaId

        }


db.create_all()
