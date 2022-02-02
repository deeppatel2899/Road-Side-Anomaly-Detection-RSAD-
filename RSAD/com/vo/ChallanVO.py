from RSAD import db
from RSAD.com.vo.UploadvideoVO import VideoVO


class ChallanVO(db.Model):
    __tablename__ = 'challanmaster'
    challanId = db.Column('challanId', db.Integer, primary_key=True, autoincrement=True)
    ownerName = db.Column('ownerName', db.String(100))
    ownerAddress = db.Column('ownerAddress', db.String(300))
    vehNumber = db.Column('vehNumber', db.String(100))
    vehType = db.Column('vehType', db.String(100))
    imagePath = db.Column('imagePath', db.String(100))
    challan_videoId = db.Column('challan_videoId', db.Integer, db.ForeignKey(VideoVO.VideoId))



    def as_dict(self):
        return {
            'ChallanId': self.challanId,
            'OwnerName': self.ownerName,
            'VehNumber':self.vehNumber,
            'VehType':self.vehType,
            'imagePath':self.imagePath,
            'challan_videoId':self.challan_videoId

        }


db.create_all()
