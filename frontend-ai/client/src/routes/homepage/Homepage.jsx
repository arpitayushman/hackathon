import { Link } from "react-router-dom";
import "./homepage.css";
import { TypeAnimation } from "react-type-animation";
import { useState,useEffect} from "react";
import { v4 as uuidv4 } from "uuid";
import { useNavigate } from "react-router-dom";
import JSZip from "jszip";

const Homepage = () => {
  const [typingStatus, setTypingStatus] = useState("human1");
  const [showModal, setShowModal] = useState(false);
  const [category, setCategory] = useState("");
  const [date, setDate] = useState("");
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const startNewChat = () => {
    const newChatId = uuidv4(); // Generate a new unique chat ID
    navigate(`/dashboard/chats/${newChatId}`); // Navigate to chat page
  
  };
  const openModal = () => {
    setShowModal(true);
  };

  const closeModal = () => {
    setShowModal(false);
  };

  const handleSubmit = async () => {
    if (!category || !date) {
      setError("Both Category and Date are required!");
      return;
    }
    setLoading(true); 
    try {
      const response = await fetch("http://localhost:5050/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ category, date }),
      });
     console.log(response);
     const responseData = await response.json();
     const csvData = responseData.files.csv?.data || [];
   // 🖼️ Extract Image Data
   const images = responseData.files.images || [];

   // Convert base64 images to URLs
   const imageUrls = {};
   images.forEach((image) => {
     if (!image.error) {
     imageUrls[image.filename] = `data:${image.mime_type};base64,${image.data}`;
     }
   });
console.log(csvData);
   navigate("/dashboard", { state: { csvData, imageUrls } });
} catch (error) {
      console.error("Error fetching or extracting ZIP:", error);
      setError("Failed to fetch prediction results");
    } finally {
      setLoading(false);
      closeModal();
    }

  };

  return (
    <div className="homepage">
            {loading && (
        <div className="loading-overlay">
          <div className="spinner"></div>
        </div>
      )}
      <img src="/orbital.png" alt="" className="orbital" />
      <div className="left">

        <h1>Runtime Terror AI</h1>
        <h2>Supercharge your creativity and productivity</h2>
        <h3>
            {/* Lorem ipsum dolor sit, amet consectetur adipisicing elit. Placeat sint
            dolorem doloribus, architecto dolor. */}
        </h3>
        <div style={{display:"flex",flexDirection:"row",justifyContent:"space-between",width:"100%"}}>
        {/* <a onClick={startNewChat}>Start New Chat</a> */}
        <a href="http://localhost:8084/">Terror AI Bot</a>
        <a onClick={openModal}>Predictive Bot</a>
        <Link to={"/stuckDashboard"}>Abnormal Transactions</Link>
        {/* <a style={{disable:true}}>what Else</a> */}
        </div>
      </div>
      {showModal && (
        <div className="modal-overlay">
          <div className="modal">
            <h2>Predictive Analysis</h2>
            <div className="modal-content">
            <label>Category:</label>
            <select value={category} onChange={(e) => setCategory(e.target.value)} className="modal-select">
              <option value="">Select Category</option>
              {/* <option value="Electricity">Electricity</option> */}
              <option value="Water">Water</option>
              <option value="Loan">Loan</option>
              <option value="Utility">Utility</option>
              <option value="Mobile">Mobile</option>
              <option value="IIFLOOOOONATD9">IIFLOOOOONATD9</option>
              <option value="Credit Card">Credit Card</option>
            </select>

            <label>Date:</label> 
            <input
              type="date"
              value={date}
              onChange={(e) => setDate(e.target.value)}
              className="modal-input"
            />
            </div>

            <div className="modal-buttons">
              <button className="btn btn-primary"  onClick={handleSubmit}>Submit</button>
              <button className="btn btn-secondary"  onClick={closeModal}>Close</button>
            </div>
          </div>
        </div>
      )}
      <div className="right">
        <div className="imgContainer">
          <div className="bgContainer">
            <div className="bg"></div>
          </div>
          <img src="/bot.png" alt="" className="bot" />
          <div className="chat">
            <img
              src={
                typingStatus === "human1"
                  ? "/human1.jpeg"
                  : typingStatus === "human2"
                  ? "/human2.jpeg"
                  : "bot.png"
              }
              alt=""
            />
            <TypeAnimation
              sequence={[
                // Same substring at the start will only be typed out once, initially
                "BD Team:I want avg Transaction count for TNEB Biller for Feb 2025",
                2000,
                () => {
                  setTypingStatus("bot");
                },
                "Terror AI:Average Transaction for Feb 2025 is 350.55",
                2000,
                () => {
                  setTypingStatus("human2");
                },
                "BD Team:I Want break up of BOUs ",
                2000,
                () => {
                  setTypingStatus("bot");
                },
                "Terror AI: sure here you go!!!",
                2000,
                () => {
                  setTypingStatus("human1");
                },
              ]}
              wrapper="span"
              repeat={Infinity}
              cursor={true}
              omitDeletionAnimation={true}
            />
          </div>
        </div>
      </div>
      {/* <div className="terms">
        <img src="/logo.png" alt="" />
        <div className="links">
          <Link to="/">Terms of Service</Link>
          <span>|</span>
          <Link to="/">Privacy Policy</Link>
        </div>
      </div> */}
    </div>
  );
};

export default Homepage;
