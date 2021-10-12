import streamlit as st
import pickle

lg = pickle.load(open('lg.pkl', 'rb'))
svc = pickle.load(open('svc.pkl', 'rb'))

def classify(num):
    if num == 0:
        return 'Negative'
    else:
        return 'Positive'
def main():
    html_temp = """
    <div style="background-color:teal ;padding:10px">
    <h2 style="color:white;text-align:center;">COVID Status Checker</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    activities=['Logistic Regression', 'SVC']
    option=st.sidebar.selectbox('Which model would you like to use?', activities)
    st.subheader(option)
    st.text('Patient symptoms:')
    cough = st.checkbox('Cough', 0, 1)
    fever = st.checkbox('Fever', 0, 1)
    sore = st.checkbox('Sore throat', 0, 1)
    short = st.checkbox('Shortness of breath', 0, 1)
    head = st.checkbox('Headache', 0, 1)
    contact = st.checkbox('Contact with confirmed', 0, 1)
    inputs = [[cough, fever, sore, short, head, contact]]

    if st.button('Check status'):
        if option == 'Logistic Regression':
            st.success(classify(lg.predict(inputs)))
        else:
            st.success(classify(svc.predict(inputs)))


if __name__ == '__main__':
    main()