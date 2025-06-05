css = """
<style>
.chat-message {
    padding: 1.5rem;
    border-radius: 10px;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
}
.chat-message.user {
    background-color: #DCF8C6;
    justify-content: flex-end;
}
.chat-message.bot {
    background-color: #E6E6E6;
    justify-content: flex-start;
}
</style>
"""

bot_template = """
<div class="chat-message bot">
    <div class="message">{{MSG}}</div>
</div>
"""

user_template = """
<div class="chat-message user">
    <div class="message">{{MSG}}</div>
</div>
"""
