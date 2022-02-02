from flask import request, render_template, redirect, url_for, session
from RSAD import app
from RSAD.com.vo.LoginVO import LoginVO
from RSAD.com.dao.LoginDAO import LoginDAO

# @app.route('/', methods=['GET'])
# def UserLoadLogin():
#     print("in login")

#     return render_template('User/Login.html')


@app.route("/validateLogin", methods=['POST'])
def validateLogin():

    loginUsername = request.form['loginUsername']
    loginPassword = request.form['loginPassword']
    # role = request.form['role']
    print(loginPassword, loginUsername)

    loginVO = LoginVO()
    loginDAO = LoginDAO()

    loginVO.loginUsername = loginUsername
    loginVO.loginPassword = loginPassword
    loginVO.loginStatus = "active"

    loginVOList = loginDAO.validateLogin(loginVO)
    print(loginVOList)
    loginDictList = [i.as_dict() for i in loginVOList]

    print(loginDictList)
    lenLoginDictList = len(loginDictList)

    if lenLoginDictList == 0:

        msg = 'Username Or Password is Incorrect !'
        return render_template('user/login.html', error=msg)

    else:

        for row1 in loginDictList:
            print(row1['loginUsername'])
            loginId = row1['loginId']

            loginUsername = row1['loginUsername']

            loginRole = row1['loginRole']

            session['session_loginId'] = loginId

            session['session_loginUsername'] = loginUsername

            session['session_loginRole'] = loginRole

            session.permanent = True

            if loginRole == 'admin':
                return redirect(url_for('adminLoadDashboard'))
            elif loginRole == 'user':
                return redirect(url_for('userLoadDashboard'))
            else:
                msg = 'Username Or Password is Incorrect !'
                return render_template('user/login.html', error=msg)


@app.route('/admin/loadDashboard', methods=['GET'])
def adminLoadDashboard():
    if LoginSession() == "admin":

        return render_template('admin/index.html')
    else:
        return redirect(url_for('LogoutSession'))


@app.route('/user/loadDashboard', methods=['GET'])
def userLoadDashboard():
    if LoginSession() == "user":

        return render_template('user/index.html',
                               loginUsername=session['session_loginUsername'])
    else:
        return redirect(url_for('LogoutSession'))


@app.route('/admin/loginSession')
def LoginSession():
    if 'session_loginId' and 'session_loginRole' in session:
        print(session)
        if session['session_loginRole'] == 'admin':

            return 'admin'

        elif session['session_loginRole'] == 'user':

            return 'user'

        print("<<<<<<<<<<<<<<<<True>>>>>>>>>>>>>>>>>>>>")

    else:

        print("<<<<<<<<<<<<<<<<False>>>>>>>>>>>>>>>>>>>>")

        return False


@app.route("/admin/logoutSession")
def LogoutSession():
    session.clear()
    return redirect(url_for('adminLoadLogin'))
