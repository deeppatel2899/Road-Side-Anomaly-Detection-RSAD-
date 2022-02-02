from flask import request, render_template, session, redirect, url_for
from RSAD import app
from werkzeug.utils import secure_filename
from RSAD.com.dao.ComplainDAO import ComplainDAO
from RSAD.com.vo.ComplainVO import ComplainVO
from RSAD.com.vo.LoginVO import LoginVO
from RSAD.com.controller.LoginController import LoginSession, LogoutSession
import os
import datetime


@app.route('/user/postComplain', methods=['POST'])
def userInsertComplain():
    try:
        if LoginSession() == 'user':

            complainVO = ComplainVO()
            complainDAO = ComplainDAO()

            complainSubject = request.form['complainSubject']
            complainDescription = request.form['complainDescription']

            try:
                complainFile = request.files['complainFile']
                print(complainFile)
                filename = secure_filename(complainFile.filename)
                filepath = "RSAD/static/Dataset/Files/"+filename
                complainFile.save(filepath)
                complainVO.complainFilepath = filepath
            except Exception as e:
                print(e)
            complainVO.complainSubject = complainSubject
            complainVO.complainDescription = complainDescription
            complainVO.complainFromLogin = session['session_loginId']
            complainDAO.insertComplain(complainVO)
            return redirect(url_for('userViewComplain'))
        else:
            return redirect(url_for('LogoutSession'))
    except Exception as ex:
        print(ex)


@app.route('/admin/viewComplain', methods=['GET'])
def adminViewComplain():
    try:
        if LoginSession() == 'admin':
            complainDAO = ComplainDAO()
            complainVOList = complainDAO.viewComplain()
            loginList = LoginVO.query.filter_by(
                loginId=complainVOList[0].complainFromLogin).first()
            complainVOList[0].complainFromLogin = loginList.loginUsername
            try:
                s = complainVOList[0].complainFilepath
                s = s.replace('RSAD', '..')
                complainVOList[0].complainFilepath = s
            except Exception as e:
                print(e)
            print("__________________", complainVOList)
            return render_template('admin/viewComplain.html', complainVOList=complainVOList)
        else:
            return redirect(url_for('LogoutSession'))
    except Exception as ex:
        print(ex)


@app.route('/user/viewComplain', methods=['GET'])
def userViewComplain():
    try:
        if LoginSession() == 'user':
            complainDAO = ComplainDAO()
            complainVO = ComplainVO()
            complainVO.complainFromLogin = session['session_loginId']
            complainVOList = complainDAO.viewComplainFrom(complainVO)
            try:
                s = complainVOList[0].complainFilepath
                s = s.replace('RSAD', '..')
                complainVOList[0].complainFilepath = s
            except Exception as e:
                print(e)
            print("__________________", complainVOList)
            return render_template('user/viewComplain.html', complainVOList=complainVOList)
        else:
            return redirect(url_for('LogoutSession'))
    except Exception as ex:
        print(ex)

@app.route('/user/viewReply', methods=['GET'])
def userViewReply():
    try:
        if LoginSession() == 'user':
            complainDAO = ComplainDAO()
            complainVO = ComplainVO()
            complainId = request.args.get('complainId')
            complainVO.complainFromLogin = session['session_loginId']
            complainVO.complainId = complainId
            complainVOList = complainDAO.viewReplayFrom(complainVO)
            loginList = LoginVO.query.filter_by(loginId=complainVOList[0].complainToLogin).first()
            complainVOList[0].complainToLogin = loginList.loginUsername
            try:
                s = complainVOList[0].replyFilepath
                s = s.replace('RSAD', '..')
                complainVOList[0].replyFilepath = s
            except Exception as e:
                print(e)
            print("__________________", complainVOList)
            return render_template('user/viewReply.html', complainVOList=complainVOList)
        else:
            return redirect(url_for('LogoutSession'))
    except Exception as ex:
        print(ex)


@app.route('/admin/deleteComplain', methods=['GET'])
def adminDeleteComplain():
    try:
        if LoginSession() == 'admin':
            complainVO = ComplainVO()
            complainDAO = ComplainDAO()
            complainId = request.args.get('complainId')
            try:
                filename = request.args.get('complainFilename')
                filename = filename.replace('..', 'RSAD/')

                os.remove(filename)
            except Exception:
                pass

            try:
                filename = request.args.get('replyFilepath')
                os.remove(filename)
            except Exception:
                pass

            complainVO.complainId = complainId
            complainDAO.deleteComplain(complainVO)

            return redirect(url_for('adminViewComplain'))
        else:
            return redirect(url_for('LogoutSession'))
    except Exception as ex:
        print(ex)


@app.route('/admin/loadReply', methods=['GET'])
def adminReplyComplain():
    try:
        if LoginSession() == 'admin':
            print()
            complainVO = ComplainVO()
            complainDAO = ComplainDAO()
            complainId = request.args.get('complainId')
            complainVO.complainId = complainId
            complainVOList = complainDAO.replyComplain(complainVO)
            print("=======complainVOList=======", complainVOList)
            return render_template('admin/replyComplain.html', complainVOList=complainVOList)
        else:
            return redirect(url_for('LogoutSession'))
    except Exception as ex:
        print(ex)


@app.route('/admin/insertReply', methods=['POST'])
def adminInsertReply():
    try:
        if LoginSession() == 'admin':
            complainId = request.form['complainId']
            replySubject = request.form['replySubject']
            replyDescription = request.form['replyDescription']
            complainVO = ComplainVO()
            complainDAO = ComplainDAO()
            complainVO.complainId = complainId

            complainVO.replySubject = replySubject
            complainVO.replyDescription = replyDescription
            complainVO.complainToLogin = session['session_loginId']
            complainVO.replyDate = datetime.datetime.now().strftime("%d-%m-%Y")
            complainVO.replyTime = datetime.datetime.now().strftime("%H:%M:%S")

            try:
                file = request.files['replyFile']
                print(file)
                filename = secure_filename(file.filename)
                filepath = "RSAD/static/Dataset/Files/"+filename
                file.save(filepath)
                complainVO.replyFilepath = filepath
            except Exception as e:
                print(e)

            complainDAO.updateComplain(complainVO)

            return redirect(url_for('adminViewComplain'))

        else:

            return redirect(url_for('LogoutSession'))

    except Exception as ex:

        print(ex)
