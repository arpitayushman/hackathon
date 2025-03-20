import { Link } from "react-router-dom";
import "./chatList.css";
import { useNavigate } from "react-router-dom";
import { v4 as uuidv4 } from "uuid";
const ChatList = () => {
  const navigate = useNavigate();

  const startNewChat = () => {
    const newChatId = uuidv4(); // Generate a new unique chat ID
    navigate(`/dashboard/chats/${newChatId}`); // Navigate to chat page
  };

  return (
    <div className="chatList">
      <span className="title">DASHBOARD</span>

      <a onClick={startNewChat}>Predicitve Analytics</a>
      {/* <hr /> */}
      {/* <span className="title">RECENT CHATS</span> */}
      {/* <div className="list"> */}

        {/* {isPending
          ? "Loading..."
          : error
          ? "Something went wrong!"
          : data?.map((chat) => (
              <Link to={`/dashboard/chats/${chat._id}`} key={chat._id}>
                {chat.title}
              </Link>
            ))} */}
      {/* </div> */}
      <hr />
      {/* <div className="upgrade">
        <img src="/logo.png" alt="" />
        <div className="texts">
          <span>Upgrade to Lama AI Pro</span>
          <span>Get unlimited access to all features</span>
        </div>
      </div> */}
    </div>
  );
};

export default ChatList;
