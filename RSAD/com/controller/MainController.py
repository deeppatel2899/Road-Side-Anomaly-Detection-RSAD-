from flask import request, render_template, redirect, url_for
from RSAD import app
from RSAD.com.controller.LoginController import LoginSession, LogoutSession


@app.route('/', methods=['GET'])
def adminLoadLogin():
    print("in login")
    return render_template('user/login.html')


@app.route('/registration', methods=['GET'])
def userLoadRegistration():
    try:

        return render_template('user/registration.html')
    except Exception as ex:
        print(ex)
#    return render_template('Admin/ManageUsers.html')



@app.route('/admin/loadArea', methods=['GET'])
def adminLoadArea():
    try:
        if LoginSession() == 'admin':
            return render_template('admin/addArea.html')
        else:
            return redirect(url_for('LogoutSession'))
    except Exception as ex:
        print(ex)


@app.route('/user/loadComplain', methods=['GET'])
def userLoadComplain():
    try:
        if LoginSession() == 'user':
            return render_template('user/postComplain.html')
        else:
            return redirect(url_for('LogoutSession'))
    except Exception as ex:
        print(ex)


@app.route('/user/loadFeedback', methods=['GET'])
def userLoadFeedback():
    try:
        if LoginSession() == 'user':
            return render_template('user/postFeedback.html')
        else:
            return redirect(url_for('LogoutSession'))
    except Exception as ex:
        print(ex)
