
from flask import request, render_template, redirect, url_for
from RSAD import app
from RSAD.com.dao.RegistrationDAO import RegistrationDAO
from RSAD.com.vo.RegistrationVO import RegistrationVO
from RSAD.com.dao.LoginDAO import LoginDAO
from RSAD.com.vo.LoginVO import LoginVO
from RSAD.com.controller.LoginController import LoginSession, LogoutSession


@app.route('/user/insertRegistration', methods=['POST'])
def userInsertRegistration():
    try:
        policestationName = request.form['policestationName']
        policestationCode = request.form['policestationCode']
        policestationAddress = request.form['policestationAddress']
        loginUsername = request.form['loginUsername']
        loginPassword = request.form['loginPassword']

        loginVO = LoginVO()
        loginDAO = LoginDAO()
        loginVO.loginUsername = loginUsername
        loginVO.loginPassword = loginPassword
        loginVO.loginRole = 'user'
        loginVO.loginStatus = 'active'
        loginDAO.insertLogin(loginVO)

        registrationVO = RegistrationVO()
        registrationDAO = RegistrationDAO()
        registrationVO.policestationName = policestationName
        registrationVO.policestationCode = policestationCode
        registrationVO.policestationAddress = policestationAddress
        loginList = LoginVO.query.filter_by(
            loginUsername=loginUsername).first()
        registrationVO.registration_loginId = loginList.loginId
        registrationDAO.insertRegistration(registrationVO)
        return render_template('user/login.html')
#        return redirect(url_for('adminViewRegistration'))
    except Exception as ex:
        return render_template('user/login.html')
        print(ex)

@app.route('/admin/viewUsers', methods=['GET'])
def adminViewRegistration():
    try:
        if LoginSession() == 'admin':
            registrationDAO = RegistrationDAO()
            registrationVOList = registrationDAO.viewRegistration()
            print("__________________", registrationVOList)
            return render_template('admin/viewUsers.html', registrationVOList=registrationVOList)
        else:
            return redirect(url_for('LogoutSession'))
    except Exception as ex:
            print(ex)


@app.route('/admin/deleteUser', methods=['GET'])
def adminDeleteRegistration():
    try:
        if LoginSession() == 'admin':
            registrationVO = RegistrationVO()
            registrationDAO = RegistrationDAO()
            loginDAO = LoginDAO()
            loginVO = LoginVO()
            registration_loginId = request.args.get('registration_loginId')

            registrationVO.registration_loginId = registration_loginId
            registrationDAO.deleteRegistration(registrationVO)

            loginVO.loginId = registration_loginId
            loginDAO.deletelogin(loginVO)

            return redirect(url_for('adminViewRegistration'))
        else:
            return redirect(url_for('LogoutSession'))
    except Exception as ex:
        print(ex)


@app.route('/admin/editUser', methods=['GET'])
def adminEditRegistration():
    try:
        if LoginSession() == 'admin':
            registrationVO = RegistrationVO()
            registrationDAO = RegistrationDAO()

            registration_loginId = request.args.get('registration_loginId')
            registrationVO.registration_loginId = registration_loginId

            registrationVOList = registrationDAO.editRegistration(
                registrationVO)

            print("=======registrationVOList=======", registrationVOList)

            print("=======type of registrationVOList=======",
                  type(registrationVOList))

            return render_template('admin/editUser.html', registrationVOList=registrationVOList)
        else:
            return redirect(url_for('LogoutSession'))
    except Exception as ex:
        print(ex)


@app.route('/admin/updateUser', methods=['POST'])
def adminUpdateRegistration():
    try:
        if LoginSession() == 'admin':
            registration_loginId = request.form['registration_loginId']
            registrationId = request.form['registrationId']

            policestationName = request.form['policestationName']
            policestationCode = request.form['policestationCode']
            policestationAddress = request.form['policestationAddress']
            username = request.form['loginUsername']
            password = request.form['loginPassword']
            registrationVO = RegistrationVO()
            registrationDAO = RegistrationDAO()

            loginDAO = LoginDAO()
            loginVO = LoginVO()
            loginVO.loginId = registration_loginId
            loginVO.loginUsername = username
            loginVO.loginPassword = password
            loginDAO.updatelogin(loginVO)

            registrationVO.registrationId = registrationId
            registrationVO.policestationName = policestationName
            registrationVO.policestationCode = policestationCode
            registrationVO.policestationAddress = policestationAddress
            registrationVO.registration_loginId = registration_loginId
            registrationDAO.updateRegistration(registrationVO)

            return redirect(url_for('adminViewRegistration'))
        else:
            return redirect(url_for('LogoutSession'))
    except Exception as ex:
        print(ex)


@app.route('/admin/blockUser', methods=['GET'])
def adminblockUser():
    try:
        if LoginSession() == 'admin':
            registration_loginId = request.args.get('registration_loginId')

            loginDAO = LoginDAO()
            loginVO = LoginVO()
            loginVO.loginId = registration_loginId
            loginDAO.blockLogin(loginVO)
            return redirect(url_for('adminViewRegistration'))
        else:
            return redirect(url_for('LogoutSession'))
    except Exception as ex:
        print(ex)


@app.route('/admin/unblockUser', methods=['GET'])
def adminunblockUser():
    try:
        if LoginSession() == 'admin':
            registration_loginId = request.args.get('registration_loginId')
            loginDAO = LoginDAO()
            loginVO = LoginVO()
            loginVO.loginId = registration_loginId
            loginDAO.unblockLogin(loginVO)
            return redirect(url_for('adminViewRegistration'))
        else:
            return redirect(url_for('LogoutSession'))
    except Exception as ex:
        print(ex)
