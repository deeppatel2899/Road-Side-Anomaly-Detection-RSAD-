from RSAD import db
import datetime

class FeedbackVO(db.Model):
    __tablename__ = 'feedbackmaster'
    feedbackId = db.Column('feedbackId', db.Integer, primary_key=True, autoincrement=True)
    feedbackSubject = db.Column('feedbackSubject', db.String(100))
    feedbackDescription = db.Column('feedbackDescription', db.String(500))
    feedbackRating = db.Column('feedbackRating', db.Integer)
    feedbackFromLogin = db.Column('feedbackFromLogin', db.Integer)
    feedbackToLogin = db.Column('feedbackToLogin', db.Integer)
    feedbackDate = db.Column('feedbackDate', db.String(100), default=datetime.datetime.now().strftime("%d-%m-%Y"))
    feedbackTime = db.Column('feedbackTime', db.String(100), default=datetime.datetime.now().strftime(("%H:%M:%S")))
    feedbackReview = db.Column('feedbackReview', db.String(100))

    def as_dict(self):
        return {
            'feedbackId': self.feedbackId,
            'feedbackSubject': self.feedbackSubject,
            'feedbackDescription': self.feedbackDescription,
            'feedbackRating': self.feedbackRating,
            'feedbackDate': self.feedbackDate,
            'feedbackTime': self.feedbackTime,
            'feedbackTologin': self.feedbackTologin,
            'feedbackFromlogin':self.feedbackFromlogin

        }


db.create_all()
