import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

# result=st.button("SIGN IN")
# st.write(result)
#



# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None

# Create a sidebar with three buttons
page = st.sidebar.radio("Select a page", ["Welcome Page","Import", "Display", "Detection","Contactus"])




def authenticate_page(username,password):
    # Authenticate the user with your authentication system
    # Return True if the credentials are valid, False otherwise
    if username == "feven" and password == "feven":
        return True
    else:
        return False

# Title for the sign-in page
# st.title("Sign In")

# Form to enter username and password in the sidebar
with st.sidebar:
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Sign In"):
        if authenticate_page(username, password):
            st.success("Logged in as {}".format(username))
        else:
            st.error("Incorrect username or password")
#
# def authenticate_page(username,password):
#     # Authenticate the user with your authentication system
#     # Return True if the credentials are valid, False otherwise
#     if username == "feven" and password == "feven":
#         return True
#     else:
#         return False
#
#     # Title for the sign-in page
#
#
# st.title("Sign In")
#
#
# # Form to enter username and password in the sidebar
# with st.sidebar:
#     username = st.text_input("Username")
#     password = st.text_input("Password", type="password")
#     if st.button("Sign In"):
#         if authenticate_page(username, password):
#             st.success("Logged in as {}".format(username))
#         else:
#             st.error("Incorrect username or password")

# # Form to enter username and password
# username = st.text_input("Username")
# password = st.text_input("Password", type="password")
#
# # Button to submit the form
# if st.button("Sign In"):
#     if authenticate_page(username, password):
#         st.success("Logged in as {}".format(username))
#     else:
#         st.error("Incorrect username or password")




def welcome_page():
    st.title("Credit Card Fraud Detector App")
    st.write(" This software application uses various techniques and algorithms to identify fraudulent credit card transactions. It is typically used by financial institutions such as banks and credit card companies to protect against fraud and unauthorized transactions. This app typically works by monitoring various aspects of each transaction, including the location of the transaction, the amount of the purchase, and the Day, Time, Distance from home and Time since last transaction of the transactions. It uses of the Symbolic Aggregate Approximation (SAX) technique to detect fraudulent credit card transactions.")

    st.title("What is Credit Card Fraud?")
    st.write("Credit card fraud refers to the unauthorized use of a credit or debit card to make purchases or withdraw cash without the consent of the cardholder. Fraudsters may steal credit card information through various means, such as phishing scams, hacking, skimming, or physical theft of the card. They may use the stolen information to make purchases online or in physical stores, transfer funds, or withdraw cash from ATMs. Credit card fraud can result in financial loss for the cardholder and damage to their credit score. It is important to report any unauthorized transactions immediately to the card issuer to minimize the potential damages.")
    st.video('C:/Users/Welcome/Downloads/How Credit Card Scams Works EMV Card Shimming Bank Fraud and Scams Credit Card Fraud.mp4')








# Define the function to display the import page
def import_page():
    st.title("Welcome To the Import Page of Credit Card Fraud Detection System Using SAX!")
    st.write("This is the page to import data.")

    # Add a file uploader to accept input data
    file = st.file_uploader("Upload a CSV file", type=["csv"])

    # Check if a file was uploaded
    if file is not None:
        # Read the data from the file
        data = pd.read_csv(file)
        st.write("Data preview:")
        st.write(data.head())

        # Store the loaded data in session state
        st.session_state.data = data


# Define the function to display the display page
# Define the function to display the display page
def display_page():
    st.title("Display")
    st.write("This is the display page.")

    # Check if data has been loaded in session state
    if st.session_state.data is not None:
        data = st.session_state.data
        st.write("Data preview:")

        # Extract the transaction amounts and timestamps
        transaction_amounts = data['Transaction Amount'].values
        timestamps = np.arange(len(transaction_amounts))
        places = data['Place'].values

        # Add a slider to select the time range to be displayed for transaction amounts
        st.write("Select the time range to be displayed for transaction amounts:")
        min_time_t, max_time_t = st.slider("Transaction Amount Time Range", 0, len(transaction_amounts), (0, len(transaction_amounts)), key="transaction_amount_slider")

        # Extract the selected time range for transaction amounts
        selected_transaction_amounts = transaction_amounts[min_time_t:max_time_t]
        selected_timestamps_t = timestamps[min_time_t:max_time_t]

        # Plot the selected time range for transaction amounts with a larger figure size
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(selected_timestamps_t, selected_transaction_amounts, label="Transaction Amount")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Transaction Amount")
        ax1.legend()
        st.pyplot(fig1)

        # Add a slider to select the time range to be displayed for place
        st.write("Select the time range to be displayed for place:")
        min_time_p, max_time_p = st.slider("Place Time Range", 0, len(places), (0, len(places)), key="place_slider")

        # Extract the selected time range for place
        selected_places = places[min_time_p:max_time_p]
        selected_timestamps_p = timestamps[min_time_p:max_time_p]

        # Plot the selected time range for place with a larger figure size
        # Define a dictionary mapping numerical values to place names
        place_names = {0: 'Abu Dhabi (0)', 1: 'Dubai (1)', 2: 'Al Ain (2)', 3: 'Sharjah (3)', 4: 'Ajman (4)', 5: 'Fujairah (5)', 6: 'Ras Al Khaimah (6)'}

        # Plot the selected time range for place with a larger figure size
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        ax2.plot(selected_timestamps_p, selected_places, label="Place")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Place")

        # Set the y-axis tick locations and labels based on the place_names dictionary
        ax2.set_yticks(list(place_names.keys()))
        ax2.set_yticklabels(list(place_names.values()))

        ax2.legend()
        st.pyplot(fig2)







        # Distance from Home (Miles)

        # Extract the transaction amounts and timestamps

        distance = data['Distance from Home (Miles)'].values
        timestamps = np.arange(len(distance))

        # Add a slider to select the time range to be displayed for transaction amounts
        st.write("Select the time range to be displayed for Distance from Home (Miles):")
        min_distance_d, max_distance_d = st.slider("Distance from Home (Miles) Range", 0, len(distance),
                                           (0, len(distance)), key="Distance from Home (Miles)_slider")

        # Extract the selected time range for transaction amounts
        selected_distance = distance[min_distance_d:max_distance_d]
        selected_distancestamps_d = timestamps[min_distance_d:max_distance_d]

        # Plot the selected time range for transaction amounts with a larger figure size
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        ax3.plot(selected_distancestamps_d, selected_distance, label="Distance from Home (Miles)")
        ax3.set_xlabel("Time")
        ax3.set_ylabel("Distance from Home (Miles")
        ax3.legend()
        st.pyplot(fig3)

        # Day

        Day = data['Day'].values
        timestamps = np.arange(len(Day))

        # Add a slider to select the time range to be displayed for place
        st.write("Select the time range to be displayed for Days:")
        min_days_d, max_days_d = st.slider("Days Time Range", 0, len(Day), (0, len(Day)), key="Days_slider")

        # Extract the selected time range for place
        selected_days = Day[min_days_d:max_days_d]
        selected_timestamps_d = timestamps[min_days_d:max_days_d]

        # Plot the selected time range for place with a larger figure size
        # Define a dictionary mapping numerical values to place names
        days_names = {0: 'Monday (0)', 1: 'Tuesday (1)', 2: 'Wednesday (2)', 3: 'Thursday (3)', 4: 'Friday (4)',
                      5: 'Saturday(5)', 6: 'Sunday (6)'}

        # Plot the selected time range for place with a larger figure size
        fig4, ax4 = plt.subplots(figsize=(12, 6))
        ax4.plot(selected_timestamps_d, selected_days, label="Day")
        ax4.set_xlabel("Time")
        ax4.set_ylabel("Days")

        # Set the y-axis tick locations and labels based on the place_names dictionary
        ax4.set_yticks(list(days_names.keys()))
        ax4.set_yticklabels(list(days_names.values()))

        ax4.legend()
        st.pyplot(fig4)


#Time Since Last Transaction (Minutes)

        times = data['Time Since Last Transaction (Minutes)'].values
        timestamps = np.arange(len(times))

        # Add a slider to select the time range to be displayed for transaction amounts
        st.write("Select the time range to be displayed for Time Since Last Transaction (Minutes):")
        min_times_t, max_times_t = st.slider("Time Since Last Transaction (Minutes)Range", 0, len(times),
                                                   (0, len(times)), key="Time Since Last Transaction (Minutes)_slider")

        # Extract the selected time range for transaction amounts
        selected_times = times[min_times_t:max_times_t]
        selected_timestamps_t = timestamps[min_times_t:max_times_t ]

        # Plot the selected time range for transaction amounts with a larger figure size
        fig5, ax5 = plt.subplots(figsize=(12, 6))
        ax5.plot(selected_timestamps_t, selected_times, label="Time Since Last Transaction (Minutes)")
        ax5.set_xlabel("Time")
        ax5.set_ylabel("Time Since Last Transaction (Minutes)")
        ax5.legend()
        st.pyplot(fig5)


#Is Fraudulent
        fraud = data['Is Fraudulent'].values
        timestamps = np.arange(len(fraud))

        # Add a slider to select the time range to be displayed for transaction amounts
        st.write("Select the time range to be displayed for Is Fraudulent:")
        min_fraud_f, max_fraud_f = st.slider("Is Fraudulent Range", 0, len(fraud),
                                             (0, len(fraud)), key="Is Fraudulent_slider")

        # Extract the selected time range for transaction amounts
        selected_times = fraud[min_fraud_f:max_fraud_f]
        selected_timestamps_t = timestamps[min_fraud_f:max_fraud_f]

        # Plot the selected time range for transaction amounts with a larger figure size
        fig6, ax6= plt.subplots(figsize=(12, 6))
        ax6.plot(selected_timestamps_t, selected_times, label="Is Fraudulent")
        ax6.set_xlabel("Time")
        ax6.set_ylabel("Is Fraudulent")
        ax6.legend()
        st.pyplot(fig6)



    else:
        st.write("No data has been imported yet.")




# Define the function to display the detection page
def detection_page():
    st.title("Detection")
    st.write("This is the detection page.")

    # Check if data has been loaded in session state
    if st.session_state.data is not None:
        data = st.session_state.data
        st.write("Data preview:")
        st.write(data.head())
        """Implements HOT-SAX."""
        import numpy as np
        from saxpy.znorm import znorm
        from saxpy.sax import sax_via_window
        from saxpy.distance import euclidean

        """Discord discovery routines."""
        import numpy as np
        from saxpy.visit_registry import VisitRegistry
        from saxpy.distance import early_abandoned_euclidean
        from saxpy.znorm import znorm

        import numpy as np
        from saxpy.hotsax import find_discords_hotsax
        from numpy import genfromtxt
        dd = genfromtxt("/Users/belsabel/Desktop/Amount22.csv", delimiter=',')
        print(dd)

        def find_discords_hotsax(series, win_size=100, num_discords=2, alphabet_size=3,
                                 paa_size=3, znorm_threshold=0.01, sax_type='unidim'):
            """HOT-SAX-driven discords discovery."""
            discords = []
            num_found = 0

            global_registry = set()

            # Z-normalized versions for every subsequence.
            znorms = np.array(
                [znorm(series[pos: pos + win_size], znorm_threshold) for pos in range(len(series) - win_size + 1)])

            # SAX words for every subsequence.
            sax_data = sax_via_window(series, win_size=win_size, paa_size=paa_size, alphabet_size=alphabet_size,
                                      nr_strategy=None, znorm_threshold=0.01, sax_type=sax_type)

            """[2.0] build the 'magic' array"""
            magic_array = list()
            for k, v in sax_data.items():
                magic_array.append((k, len(v)))

            """[2.1] sort it ascending by the number of occurrences"""
            magic_array = sorted(magic_array, key=lambda tup: tup[1])

            while num_found < num_discords:

                best_discord = find_best_discord_hotsax(series, win_size, global_registry, sax_data, magic_array,
                                                        znorms)

                if -1 == best_discord[0]:
                    break

                discords.append(best_discord)

                mark_start = max(0, best_discord[0] - win_size + 1)
                mark_end = best_discord[0] + win_size

                for i in range(mark_start, mark_end):
                    global_registry.add(i)

                num_found += 1

            # Return all discords found during the iterations
            return discords

        def find_best_discord_hotsax(series, win_size, global_registry, sax_data, magic_array, znorms):
            """Find the best discord with hotsax."""

            """[3.0] define the key vars"""
            best_so_far_position = -1
            best_so_far_distance = 0.

            distance_calls = 0

            visit_array = np.zeros(len(series), dtype=np.int)

            """[4.0] and we are off iterating over the magic array entries"""
            for entry in magic_array:

                """[5.0] current SAX words and the number of other sequences mapping to the same SAX word."""
                curr_word = entry[0]
                occurrences = sax_data[curr_word]

                """[6.0] jumping around by the same word occurrences makes it easier to
                nail down the possibly small distance value -- so we can be efficient
                and all that..."""
                for curr_pos in occurrences:

                    if curr_pos in global_registry:
                        continue

                    """[7.0] we don't want an overlapping subsequence"""
                    mark_start = curr_pos - win_size + 1
                    mark_end = curr_pos + win_size
                    visit_set = set(range(mark_start, mark_end))

                    """[8.0] here is our subsequence in question"""
                    cur_seq = znorms[curr_pos]

                    """[9.0] let's see what is NN distance"""
                    nn_dist = np.inf
                    do_random_search = True

                    """[10.0] ordered by occurrences search first"""
                    for next_pos in occurrences:

                        """[11.0] skip bad pos"""
                        if next_pos in visit_set:
                            continue
                        else:
                            visit_set.add(next_pos)

                        """[12.0] distance we compute"""

                        dist = euclidean(cur_seq, znorms[next_pos])
                        distance_calls += 1

                        """[13.0] keep the books up-to-date"""
                        if dist < nn_dist:
                            nn_dist = dist
                        if dist < best_so_far_distance:
                            do_random_search = False
                            break

                    """[13.0] if not broken above,
                    we shall proceed with random search"""
                    if do_random_search:
                        """[14.0] build that random visit order array"""
                        curr_idx = 0
                        for i in range(0, (len(series) - win_size + 1)):
                            if not (i in visit_set):
                                visit_array[curr_idx] = i
                                curr_idx += 1
                        it_order = np.random.permutation(visit_array[0:curr_idx])
                        curr_idx -= 1

                        """[15.0] and go random"""
                        while curr_idx >= 0:
                            rand_pos = it_order[curr_idx]
                            curr_idx -= 1

                            dist = euclidean(cur_seq, znorms[rand_pos])
                            distance_calls += 1

                            """[16.0] keep the books up-to-date again"""
                            if dist < nn_dist:
                                nn_dist = dist
                            if dist < best_so_far_distance:
                                nn_dist = dist
                                break

                    """[17.0] and BIGGER books"""
                    if (nn_dist > best_so_far_distance) and (nn_dist < np.inf):
                        best_so_far_distance = nn_dist
                        best_so_far_position = curr_pos

            return best_so_far_position, best_so_far_distance

        discords = find_discords_hotsax(dd[0:5000], 100, 500)
        discords

        import matplotlib.pyplot as plt

        def plot_discords(series, discords, win_size=100, discord_color='red', figsize=(10, 6)):
            """Plot the discords (anomalies) in the series."""
            plt.figure(figsize=figsize)  # Set the size of the plot
            for discord in discords:
                discord_start = max(0, discord[0] - win_size + 1)
                discord_end = discord[0] + win_size
                plt.plot(range(discord_start, discord_end), series[discord_start:discord_end], color=discord_color)
            plt.plot(range(len(series)), series, label='Series')
            plt.legend()
            plt.xlabel('Index')
            plt.ylabel('Value')
            plt.title('Discords (Anomalies) in the Series')
            plt.show()

        # Assuming you have a time series 'series' and the discords found using 'find_discords_hotsax' function stored in 'discords'
        # You can call the 'plot_discords' function as follows:
        plot_discords(dd, discords, win_size=100, discord_color='red', figsize=(10, 6))  # Adjust figsize as needed


    else:
        st.write("No data has been imported yet.")


def contact_page():

    st.header(":mailbox: Get In Touch With Me!")

    contact_form = """
    <form action="https://formsubmit.co/fevenbelay123@gmail.com" method="POST">
         <input type="hidden" name="_captcha" value="false">
         <input type="text" name="name" placeholder="Your name" required>
         <input type="email" name="email" placeholder="Your email" required>
         <textarea name="message" placeholder="Your message here"></textarea>
         <button type="submit">Send</button>
    </form>
    """

    st.markdown(contact_form, unsafe_allow_html=True)

    # Use Local CSS File
    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    local_css("style.css")





    # Title for the feedback page
    st.title("Feedback")

    # Add radio buttons for like and dislike options
    feedback = st.radio("Did you like this app?", ("Like", "Dislike"))

    # Display appropriate message based on the user's feedback
    if feedback == "Like":
        st.write("Thank you for your positive feedback!")
    else:
        st.write("We're sorry to hear that. Please let us know how we can improve.")


# Depending on the button selected in the sidebar, display the corresponding page
if page == "authenticate":
    authenticate_page("feven","feven")
elif page == "Import":
    import_page()
elif page == "Display":
    display_page()
elif page == "Detection":
    detection_page()
elif page == "Welcome Page" :
    welcome_page()
elif page == "Contactus":
    contact_page()
