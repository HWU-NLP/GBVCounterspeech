import streamlit as st

st.write("# GBV Counterspeech Task Guideline")

with st.sidebar:
    st.warning("""
        💡 Please **read the annotation guidelines** on this page carefully.  
        \n\n 💡 After you move to <Start Annotation Task> page, you need to **pass a qualification test** before starting the annotation task.  
        \n\n 💡 After the start, we also provide **Task Overview** and **GBV Counterspeech Annotation Guidelines** in the sidebar for reference.
    """)

    st.markdown("<br>", unsafe_allow_html=True)
    
with st.container(border=True):
    st.markdown("""
    :blue-background[**Key Concepts**]
    * **Gender-Based Violence (GBV)** is a complex and multifaceted issue that includes hybrid behaviours of physical, digital, verbal, psychological, and sexual violence. It can take both implicit and explicit forms and often occurs across multiple spaces and contexts. GBV contains various forms of abuse and specialist focuses, such as coercive control, domestic violence, intimate partner violence, sexual harassment, and stalking.
    * **Counterspeech (CS):** A direct response to challenge or counter hateful or harmful speech.
    """)

st.markdown("In this task, you will see pairs of GBV text, and the counterspeech written in response to it. For each of these pairs, you need to look at the counterspeech and **assign labels**, and give the **feedback**.")
st.markdown("1️⃣ **CS Strategy**  \n You need to label what kind of strategy was used to counter GBV text. For example, is the response humorous and sarcastic? There are eight options: Empathy and Affiliation, Warning of Consequence, Hypocrisy or Contradiction, Shaming or Labelling, Denouncing, Providing Facts, Humour or Sarcasm, and Questioning. You may select **up to 3 strategies** if needed.")
st.markdown("*️⃣ **Feedback** is also welcome after annotating each pair to see if any issues you meet during the annotation process.")
st.markdown("Our detailed instructions below give you helpful hints and examples in terms of the language you should look for to assign each label.")
st.markdown("<br>", unsafe_allow_html=True)

st.markdown("##### CS Strategy Annotation Guideline")
st.markdown(
    """
    <style>
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #f4f4f4;
            font-weight: bold;
        }
        .bold { font-weight: bold; }
        .italic { font-style: italic; }
        .underline { text-decoration: underline; }
        .yellow { color: #D4A017; } /* Golden yellow for tone label */
    </style>

    <table>
        <colgroup>
            <col style="width: 25%;">
            <col style="width: 75%;">
        </colgroup>
        <tr>
            <th>Strategy</th>
            <th>Definition</th>
        </tr>
        <tr>
            <td><b>Empathy and Affiliation</b></td>
            <td>
                <span class="bold">Focuses on promoting understanding</span>, fostering peace and finding common ground.<br>
                <span class="yellow">📣 Tone:</span> Kind, compassionate, understanding language.<br>
                <span class="bold italic underline">Example</span><br>
                <span class="italic">These are people in need—our brothers and sisters. We should help them.</span>
            </td>
        </tr>
        <tr>
            <td><b>Warning of Consequence</b></td>
            <td>
                <span class="bold">Cautioning the speaker</span> about the impact of their words via potential negative outcomes, such as legal, social, or personal consequences.<br>
                <span class="yellow">📣 Tone:</span> Serious, cautionary or urgent.<br>
                <span class="bold italic underline">Example</span><br>
                <span class="italic">Ignoring history risks hurting future generations by not teaching them the lessons of the past.</span>
            </td>
        </tr>
        <tr>
            <td><b>Hypocrisy or Contradiction</b></td>
            <td>
                <span class="bold">Discredit the argument</span> through critical analysis by pointing out inconsistencies, illogical reasoning, contradictions, or double standards.<br>
                <span class="yellow">📣 Tone:</span> Critical, logical, analytical.<br>
                <span class="bold italic underline">Example</span><br>
                <span class="italic">
                <b>GBV:</b> Zionist and Jewish control of America is the main cause of global terrorism and conflict.<br>
                <b>CS:</b> Isn't colonialism and Western intervention abroad what led to much of the extremism?
                </span>
            </td>
        </tr>
        <tr>
            <td><b>Shaming or Labelling</b></td>
            <td>
                <span class="bold">Direct and confrontational, including personal attacks</span>. Attacks the speaker by using negative labels or terms to emphasise the inappropriateness of their behaviour.<br>
                <span class="yellow">📣 Tone:</span> Accusatory, confrontational, disapproving.<br>
                <span class="bold italic underline">Example</span><br>
                <span class="italic">It seems you've become intolerant to homosexuals, maybe reflect on why love bothers you so much.</span>
            </td>
        </tr>
        <tr>
            <td><b>Denouncing</b></td>
            <td>
                <span class="bold">Refuting without personal attacks</span>. Condemns or rejects the hateful views, stating that they are wrong, unacceptable, harmful, etc.<br>
                <span class="yellow">📣 Tone:</span> Firm, direct, authoritative.<br>
                <span class="bold italic underline">Example</span><br>
                <span class="italic"><b>GBV:</b> I miss the good old days when we had black domestic workers.<br>
                <b>CS:</b> The West got rid of slavery. But we've never really left racism.</span>
            </td>
        </tr>
        <tr>
            <td><b>Providing Facts</b></td>
            <td>
                <span class="bold">Correct misinformation and false narratives</span>. Uses facts, statistics, or evidence to counter false claims.<br>
                <span class="yellow">📣 Tone:</span> Factual, corrective, neutral.<br>
                <span class="bold italic underline">Example</span><br>
                <span class="italic"><b>GBV:</b> Women can't handle money.<br>
                <b>CS:</b> What proof do you have? This stereotype comes from medieval myths, not real financial behavior.</span>
            </td>
        </tr>
        <tr>
            <td><b>Humour or Sarcasm</b></td>
            <td>
                <span class="bold">Lighthearted, diminishes the comment by ridiculing it</span> (or the perpetrator) via humour, irony, or sarcasm.<br>
                <span class="yellow">📣 Tone:</span> Playful, humorous, mocking.<br>
                <span class="bold italic underline">Example</span><br>
                <span class="italic"><b>GBV:</b> Jews are why the World Cup is so expensive.<br>
                <b>CS:</b> Oh, I thought it was players' salaries... but sure, let's blame the usual Jewish conspiracy!</span>
            </td>
        </tr>
        <tr>
            <td><b>Questioning</b></td>
            <td>
                <span class="bold">Exposes flaws in the argument</span> by probing its assumptions, logic, or intentions, often through asking pointed or clarifying questions. <br>
                <span class="yellow">📣 Tone:</span> Inquisitive, probing, reflective.<br>
                <span class="bold italic underline">Example</span><br>
                <span class="italic"><b>GBV:</b> All Muslims are incompatible with any Western society.<br>
                <b>CS:</b> I'm not sure what you are trying to say by incompatible. What do you mean by incompatible?</span>
            </td>
        </tr>
    </table>
    """,
    unsafe_allow_html=True
)
st.markdown("<br>", unsafe_allow_html=True)

st.markdown("##### Feedback")
st.markdown("""
Please see if any of the following issues exist in the GBV and counterspeech pairs, and comment more in the textbox if needed.
* **STRATEGY CONFUSION:** Hard to choose the correct strategies for this example
* **MISMATCH:** CS refers to a completely different subject.  \n :green[_E.g. HS related to **race**, CS related to **feminism**_]
* **PARTIAL MATCH / INDIRECT:** CS sort of addresses the issue, but not in a straightforward way.  \n :green[_E.g. HS is against **women**, but CS uses words like **feminism**_]
* **NOT PERSUASIVE:** CS addresses the issue, but I didn't find it very convincing
* **OTHER:** CS is not good for other reasons, such as being uninformative, vague, ambiguous.  \n :green[_E.g. an uninformative CS response such as **'Why do you think that way?'** without any further text_]
""")
st.markdown("<br>", unsafe_allow_html=True)
