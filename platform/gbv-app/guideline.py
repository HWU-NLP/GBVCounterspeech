import streamlit as st

st.write("# GBV Annotation Task Guideline")

with st.sidebar:
    st.warning("""
        💡 Please **read the annotation guidelines** on this page carefully.  
        \n\n 💡 After you move to <Start Annotation Task> page, you need to **pass a qualification test** before starting the annotation task.  
        \n\n 💡 After the start, we also provide **Task Overview** and **GBV Annotation Guidelines** in the sidebar for reference.
    """)

    st.markdown("<br>", unsafe_allow_html=True)
    
with st.container(border=True):
    st.markdown(f":blue-background[**Key Concepts of Gender-Based Violence (GBV)**] \n\n A complex and multifaceted issue that includes hybrid behaviours of physical, digital, verbal, psychological, and sexual violence. It can take both implicit and explicit forms and often occurs across multiple spaces and contexts. GBV contains various forms of abuse and specialist focuses, such as coercive control, domestic violence, intimate partner violence, sexual harassment, and stalking.")


st.markdown("For each GBV text, two subtasks need to be annotated: GBV Target, and GBV Form.")
st.markdown("1️⃣ **Subtask 1 - GBV Target**  \n Is there a particular sub-category used to refer to the **woman/women or ideology being targeted**? Please highlight the relevant text.  \n It could be _'a breastfeeding woman, a girl, feminist, feminism, gender equality, etc.'._")
st.markdown("2️⃣ **Subtask 2 - GBV Form**  \n The type of GBV expressed in the text. Here are five options: Dehumanisation, Threatening, Derogation, Animosity, and Support of Hate Crimes. You may select **up to two forms** if needed.")
st.markdown("*️⃣ **Feedback** is also welcome after annotating each text to see if any issues you meet during the annotation process.")
st.markdown("Our detailed instructions below give you helpful hints and examples in terms of the language you should look for to assign each label.")
st.markdown("<br>", unsafe_allow_html=True)


st.markdown("##### GBV Form Annotation Guideline")
st.markdown("Below are the five types of GBV forms with definitions and examples.")
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
            <col style="width: 20%;">
            <col style="width: 80%;">
        </colgroup>
        <tr>
            <th>Form</th>
            <th>Definition</th>
        </tr>
        <tr>
            <td><b>Dehumanisation</b></td>
            <td>
                <span class="bold">Reduces the target</span> to a <b>subhuman or non-human</b> status.<br>
                <span class="yellow">💡 Hint:</span> comparisons to animals, objects, or entities, stripping away their humanity.<br>
                <span class="bold italic underline">Example</span><br>
                <span class="italic">Women are pigs.</span>
            </td>
        </tr>
        <tr>
            <td><b>Threatening</b></td>
            <td>
                <span class="bold">Explicit, direct (to target), threatening language or incitement to harm.</span> Expresses intent or encourages others to <b>take action against the target</b>.<br>
                <span class="yellow">💡 Hint:</span> threats of physical, emotional, or privacy-related harm, direct threats or calls for harm, such as violence or violation.<br>
                <span class="bold italic underline">Example</span><br>
                <span class="italic">These women should be shot? Agreed!</span>
            </td>
        </tr>
        <tr>
            <td><b>Derogation</b></td>
            <td>
                <span class="bold">Explicit derogatory, insulting, or demeaning</span> language, focusing on the target's <b>character, abilities, or physical attributes</b>.<br>
                <span class="yellow">💡 Hint:</span> negative stereotypes, insults, or slurs.<br>
                <span class="bold italic underline">Example</span><br>
                <span class="italic">Women's football is so slow and clumsy.</span>
            </td>
        </tr>
        <tr>
            <td><b>Animosity</b></td>
            <td>
                <span class="bold">Implicit or subtle hostility,</span> often framed in a way that normalizes or downplays sexism via statements such as <b>backhanded compliments</b>.<br>
                <span class="yellow">💡 Hint:</span> indirect insults or subtle forms of bias.<br>
                <span class="bold italic underline">Example</span><br>
                <span class="italic">Women are delicate flowers who need to be cherished.</span>
            </td>
        </tr>
        <tr>
            <td><b>Support of Hate Crimes</b></td>
            <td>
                <span class="bold">Glorifies, supports, or justifies hate crimes or discrimination.</span> Praise for violent actions, systemic discrimination, or organizations that perpetuate hate.<br>
                <span class="yellow">💡 Hint:</span> expressions that endorse, justify, or deny hate-based violence or discrimination.<br>
                <span class="bold italic underline">Example</span><br>
                <span class="italic">Women shouldn't show that much skin; it's their own fault if they get raped.</span>
            </td>
        </tr>
    </table>
    """,
    unsafe_allow_html=True
)
st.markdown("**Key points between GBV forms:**")
st.markdown("""🔸 Animosity vs. Derogation:
* Animosity: Implicit hostility (e.g., backhanded compliments, veiled criticism).
* Derogation: Explicit insults or demeaning language.
""")
st.markdown("""🔸 Support of Hate Crimes vs. Threatening:
* Support of Hate Crimes: Generalised statements endorsing violence or harm.
* Threatening: Specific threats targeting an individual or group.
""")
st.markdown("When in doubt between forms: If you are unsure whether a GBV text fits one form or another, you may **select both**.")
st.markdown("<br>", unsafe_allow_html=True)

st.markdown("##### Examples of Annotating GBV Text")
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
            <col style="width: 35%;">
            <col style="width: 65%;">
        </colgroup>
        <tr>
            <th>GBV Text</th>
            <th>Annotation Example</th>
        </tr>
        <tr>
            <td>Women should be shot? Agreed!</td>
            <td>
                <span class="bold underline">GBV Target:</span> women<br>
                <span class="bold underline">GBV Form:</span> Threatening<br>
            </td>
        </tr>
        <tr>
            <td>I don't want to have to be around a breastfeeding woman, if she wants to do that she can do it at home.</td>
            <td>
                <span class="bold underline">GBV Target:</span> a breastfeeding woman<br>
                <span class="bold underline">GBV Form:</span> Animosity<br>
            </td>
        </tr>
        <tr>
            <td>Women's football is so slow and clumsy.</td>
            <td>
                <span class="bold underline">GBV Target:</span> women<br>
                <span class="bold underline">GBV Form:</span> Derogation<br>
            </td>
        </tr>
        <tr>
            <td>Women are pigs.</td>
            <td>
                <span class="bold underline">GBV Target:</span> women<br>
                <span class="bold underline">GBV Form:</span> Dehumanisation<br>
            </td>
        </tr>
    </table>
    """,
    unsafe_allow_html=True
)
st.markdown("<br>", unsafe_allow_html=True)

st.markdown("##### Feedback")
st.markdown("""
Please see if any of the following issues exist in this annotation task, and comment more in the textbox if needed.
* **TARGET CONFUSION:** Hard to select GBV targets for this example
* **FORM CONFUSION:** Hard to select GBV forms for this example
* **UNCOMFORTABLE:** I do not feel good when I see the example
""")
