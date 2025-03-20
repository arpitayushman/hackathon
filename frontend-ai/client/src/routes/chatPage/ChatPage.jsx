import "./chatPage.css";
import NewPrompt from "../../components/newPrompt/NewPrompt";
import { useQuery } from "@tanstack/react-query";
import { useLocation ,useParams} from "react-router-dom";
import Markdown from "react-markdown";
import { useEffect,useState } from "react";


const ChatPage = () => {
  // const { state } = useLocation(); // Get message from navigation state
  // const { id } = useParams();

  // const [apiId, setApiId] = useState(null);
  // useEffect(() => {
  //   console.log("inside use effect")
  //   if(!state.message){
  //     return;
  //   }
  //   // Check sessionStorage for API-generated chat ID
  //   const fetchSQLData = async () => {
  //   try {
  //     // Fetch API chat ID asynchronously
  //     console.log
  //     const response = await fetch(
  //       `http://localhost:5000/api/v0/generate_sql?question=${encodeURIComponent(state.message)}`,
  //       { method: "GET", credentials: "include" }
  //     );
  //     console.log(response);
  //     if (!response.ok) throw new Error("Failed to generate chat ID");
  
  //     const data = await response.json();
  //     console.log(data)
  //     const {id} = data;
  //     console.log(id);
  
  //     // Store API ID in session storage (ChatPage will pick it up)
  //     setApiId(id);
  //   } catch (error) {
  //     console.error("Error fetching chat ID:", error);
  //   }
  // }
  // fetchSQLData();
  // }, [id]);



  return (
    <div className="chatPage">

      <div className="wrapper">
        <div className="chat">
 
        {/* {isPending ? (
          "Loading..."
        ) : error ? (
          "Something went wrong!"
        ) : (
          <div className="message user">
            <Markdown>{data?.question}</Markdown> {/* Show only the question */}
          {/* </div>
        )} */} 
                  {/* {state?.message && (
            <div className="message user">
              <Markdown>{state.message}</Markdown>
            </div>
          )} */}
     <NewPrompt  />
        </div>
      
      </div>
     
    </div>
  );
};

export default ChatPage;
