import { useEffect, useRef, useState } from "react";
import "./newPrompt.css";
import model from "../../lib/gemini";
import Markdown from "react-markdown";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import * as XLSX from "xlsx"; 
import Plot from "react-plotly.js";

const NewPrompt = ({ chatId,message }) => {
  const [typingText, setTypingText] = useState("");
  const [question, setQuestion] = useState(message);
  const [chatHistory, setChatHistory] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  // const [answer, setAnswer] = useState("");
  // const [tableData, setTableData] = useState([]); 
  // const [headers, setHeaders] = useState([]);
  // const [plotData, setPlotData] = useState(null);
  // const [followUpQuestions, setFollowUpQuestions] = useState([]);
  const [chatID,setChatID]=useState(chatId);



  // const chat = model.startChat({
  //   history: [
  //     data?.history.map(({ role, parts }) => ({
  //       role,
  //       parts: [{ text: parts[0].text }],
  //     })),
  //   ],
  //   generationConfig: {
  //     // maxOutputTokens: 100,
  //   },
  // });

  const endRef = useRef(null);
  const formRef = useRef(null);

  useEffect(() => {
    if (endRef.current) {
      endRef.current.scrollIntoView({ behavior: "smooth",block:"end" });
    }
  }, [chatHistory]);

  const queryClient = useQueryClient();

  // const mutation = useMutation({
  //   mutationFn: () => {
  //     return fetch(`${import.meta.env.VITE_API_URL}/api/chats/${data._id}`, {
  //       method: "PUT",
  //       credentials: "include",
  //       headers: {
  //         "Content-Type": "application/json",
  //       },
  //       body: JSON.stringify({
  //         question: question.length ? question : undefined,
  //         answer,
        
  //       }),
  //     }).then((res) => res.json());
  //   },
  //   onSuccess: () => {
  //     queryClient
  //       .invalidateQueries({ queryKey: ["chat", data._id] })
  //       .then(() => {
  //         formRef.current.reset();
  //         setQuestion("");
  //         setAnswer("");
      
  //       });
  //   },
  //   onError: (err) => {
  //     console.log(err);
  //   },
  // });

  // const add = async (text, isInitial) => {
  //   if (!isInitial) setQuestion(text);

  //   try {
  //     const result = await chat.sendMessageStream([text]);
  //     let accumulatedText = "";
  //     for await (const chunk of result.stream) {
  //       const chunkText = chunk.text();
  //       console.log(chunkText);
  //       accumulatedText += chunkText;
  //       setAnswer(accumulatedText);
  //     }

  //     mutation.mutate();
  //   } catch (err) {
  //     console.log(err);
  //   }
  // };
  // useEffect(() => {
  //   if (!chatID) return;
  //   console.log("inside use effect");
  
  //   const fetchSQLData = async () => {
  //     try {
  //       const response = await fetch(`http://localhost:8084/api/v0/run_sql?id=${chatID}`);
  //       if (!response.ok) throw new Error("Failed to fetch SQL data");
  //       console.log(response);
  
  //       const jsonData = await response.json();
  //       console.log("SQL Response:", jsonData);
  
  //       // ✅ Parse the 'df' field (it's a JSON string)
  //       const parsedData = JSON.parse(jsonData.df); // Extract and parse 'df'
  
  //       if (parsedData.length > 0) {
  //         setHeaders(Object.keys(parsedData[0])); // Extract dynamic column names
  //       }
  
  //       setTableData(parsedData);
  //       fetchPlotlyFigure();
  //     } catch (error) {
  //       console.error("Error fetching SQL data:", error);
  //     }
  //   };
  //   const fetchPlotlyFigure = async () => {
  //     try {
  //       console.log("Fetching Plotly figure...");
  //       const response = await fetch(`http://localhost:8084/api/v0/generate_plotly_figure?id=${chatID}`);
  //       if (!response.ok) throw new Error("Failed to fetch plotly figure");

  //       const plotJson = await response.json();
  //       setPlotData(JSON.parse(plotJson.fig));
  //       fetchFollowUpQuestions(); // Parse and store figure
  //     } catch (error) {
  //       console.error("Error fetching Plotly figure:", error);
  //     }
  //   };
  //   const fetchFollowUpQuestions = async () => {
  //     try {
  //       console.log("Fetching follow-up questions...");
  //       const response = await fetch(`http://localhost:8084/api/v0/generate_followup_questions?id=${chatID}`);
  //       if (!response.ok) throw new Error("Failed to fetch follow-up questions");
  
  //       const followUpJson = await response.json();
  //       setFollowUpQuestions(followUpJson.questions || []);
  //     } catch (error) {
  //       console.error("Error fetching follow-up questions:", error);
  //     }
  //   };
  
  //   fetchSQLData();
  // }, [chatID]); 
  // useEffect(() => {
  //   if (!chatID) return;
    
  //   const fetchData = async () => {
  //     try {
  //       const sqlResponse = await fetch(`http://localhost:8084/api/v0/run_sql?id=${chatID}`);
  //       if (!sqlResponse.ok) throw new Error("Failed to fetch SQL data");
  //       const sqlData = await sqlResponse.json();
  //       const parsedData = JSON.parse(sqlData.df);
  //       const headers = parsedData.length > 0 ? Object.keys(parsedData[0]) : [];

  //       const plotResponse = await fetch(`http://localhost:8084/api/v0/generate_plotly_figure?id=${chatID}`);
  //       const plotJson = await plotResponse.json();
  //       const plotData = JSON.parse(plotJson.fig);

  //       const followUpResponse = await fetch(`http://localhost:8084/api/v0/generate_followup_questions?id=${chatID}`);
  //       const followUpJson = await followUpResponse.json();

  //       setChatHistory(prevHistory => [
  //         ...prevHistory,
  //         { question: typingText, tableData: parsedData, headers, plotData, followUpQuestions: followUpJson.questions || [] }
  //       ]);
  //     } catch (error) {
  //       console.error("Error fetching data:", error);
  //     }
  //   };

  //   fetchData();
  // }, [chatID]);
  // const handleExcelDownload = () => {
  //   if (tableData.length === 0) return;

  //   const worksheet = XLSX.utils.json_to_sheet(tableData);
  //   const workbook = XLSX.utils.book_new();
  //   XLSX.utils.book_append_sheet(workbook, worksheet, "Sheet1");

  //   XLSX.writeFile(workbook, `table_data_${chatId}.xlsx`);
  // };
  const handleExcelDownload = (tableData, chatId) => {
    if (tableData.length === 0) return;
    const worksheet = XLSX.utils.json_to_sheet(tableData);
    const workbook = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(workbook, worksheet, "Sheet1");
    XLSX.writeFile(workbook, `table_data_${chatId}.xlsx`);
  };
  const handleInputChange = (e) => {
    setTypingText(e.target.value);
  };
  const handleFollowUpClick = (question) => {
    setTypingText(question); // Set the question as input
    setIsLoading(true);
    setTimeout(() => {
      handleSubmit(question); // Trigger the existing submit function
    }, 100);
  };

const handleSubmit = async (eOrText) => {
  let text = "";

  // Check if a text string is passed (from follow-up click)
  if (typeof eOrText === "string") {
    text = eOrText.trim();
  } else {
    eOrText.preventDefault();
    text = typingText.trim();
  }
    console.log(text);
    if (!text || isLoading) return;
    setIsLoading(true);
    try{
      const response=await fetch(`http://localhost:8084/api/v0/generate_sql?question=${encodeURIComponent(text)}`,{method:"GET",credentials:"include"});
      console.log("what happening"+response);
      if (!response.ok) {
        throw new Error("Failed to generate SQL query");
      }
      const data = await response.json();
      const {id} = data; 
setChatID(id);
const fetchData = async () => {
  try {
    const sqlResponse = await fetch(`http://localhost:8084/api/v0/run_sql?id=${id}`);
    if (!sqlResponse.ok) throw new Error("Failed to fetch SQL data");
    const sqlData = await sqlResponse.json();
    const parsedData = JSON.parse(sqlData.df);
    const headers = parsedData.length > 0 ? Object.keys(parsedData[0]) : [];

    const plotResponse = await fetch(`http://localhost:8084/api/v0/generate_plotly_figure?id=${id}`);
    const plotJson = await plotResponse.json();
    const plotData = JSON.parse(plotJson.fig);

    const followUpResponse = await fetch(`http://localhost:8084/api/v0/generate_followup_questions?id=${id}`);
    const followUpJson = await followUpResponse.json();

    // Now update chat history with complete response
    setChatHistory(prevHistory => [
      ...prevHistory,
      { question: text, tableData: parsedData, headers, plotData, followUpQuestions: followUpJson.questions || [] }
    ]);
    
  } catch (error) {
    console.error("Error fetching data:", error);
  }finally{
  
      setTypingText(""); 
      setIsLoading(false); 
    
  }
};

fetchData();

} catch (error) {
console.error("Error fetching SQL generation:", error);
}


    };

  // IN PRODUCTION WE DON'T NEED IT
  const hasRun = useRef(false);

  // useEffect(() => {
  //   if (!hasRun.current) {
  //     if (data?.history?.length === 1) {
  //       add(data.history[0].parts[0].text, true);
  //     }
  //   }
  //   hasRun.current = true;
  // }, []);

  return (
    <>
          {/* Live question update (only while typing) */}

          {chatHistory.map((chat, index) => (
        <div key={index} className="chat-container">
      
          <div className="message user">{chat.question}</div>
          <div className="message">
            {chat.tableData.length > 0 && (
              <div className="table-container">
                <h3 style={{marginBottom:"10px"}}>Result</h3>
                <table>
                  <thead>
                    <tr>
                      {chat.headers.map((header, idx) => (
                        <th key={idx}>{header.replace(/_/g, " ").toUpperCase()}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {chat.tableData.map((row, rowIndex) => (
                      <tr key={rowIndex}>
                        {chat.headers.map((header, colIndex) => (
                          <td key={colIndex}>{row[header]}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
                <button onClick={() => handleExcelDownload(chat.tableData, chatID)} className="download-btn">Download CSV</button>
              </div>
            )}

            {chat.plotData && (
              <div>
                <h3 style={{marginBottom:"10px"}}>Generated Chart</h3>
                <Plot data={chat.plotData.data} layout={chat.plotData.layout} />
              </div>
            )}

            {chat.followUpQuestions.length > 0 && (
              <div className="follow-up-questions">
                <h3 style={{marginBottom:"10px"}}>Follow-up Questions</h3>
                <div className="follow-up-container">
                  {chat.followUpQuestions.map((q, idx) => (
                 <button className="follow-up-btn" onClick={() => handleFollowUpClick(q)}>{q}</button>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      ))}
          {typingText && <div className="message user typing">{typingText}</div>
              
          
          }
                {isLoading && (
                      <div className="loader-container message">
                        <div className="spinner"></div>
                        <p>Thinking...</p>
                      </div>
                    )}
 <div ref={endRef} style={{ paddingBottom: "190px" }}></div>
      <form className="newForm" onSubmit={handleSubmit} ref={formRef} >
        <input type="text" name="text" placeholder="Ask anything..." value={typingText} onChange={handleInputChange} disabled={isLoading} />
        <button disabled={isLoading}>
          <img src="/arrow.png" alt="" />
        </button>
      </form>


{/*     
      {question && <div className="message user">{question}</div>}
      {answer && (
        <div className="message">
          <Markdown>{answer}</Markdown>
        </div>
      )}
          
            {tableData.length > 0 && (
  <div className="table-container">
    <h3 style={{marginBottom:"20px"}}>Result</h3>
    <table>
      <thead>
        <tr>
          {headers.map((header, index) => (
            <th key={index}>{header.replace(/_/g, " ").toUpperCase()}</th> 
          ))}
        </tr>
      </thead>
      <tbody>
        {tableData.map((row, rowIndex) => (
          <tr key={rowIndex}>
            {headers.map((header, colIndex) => (
              <td key={colIndex}>
                {typeof row[header] === "number" && String(row[header]).length === 13 
                  ? new Date(row[header]).toLocaleDateString() 
                  : row[header]}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
    <button className="download-btn" onClick={handleExcelDownload}>
            Download CSV
          </button>

          {plotData && (
        <div>
          <h3>Generated Chart</h3>
          <Plot data={plotData.data} layout={plotData.layout} />
        </div>
      )}
      {followUpQuestions.length > 0 && (
      <div className="follow-up-questions">
        <h3>Follow-up Questions</h3>
        <ul>
          {followUpQuestions.map((question, index) => (
            <li key={index}>{question}</li>
          ))}
        </ul>
      </div>
    )}
  </div>
)}
      <div className="endChat" ref={endRef}></div>
      <form className="newForm" onSubmit={handleSubmit} ref={formRef}>

        <input type="text" name="text" placeholder="Ask anything..." />
        <button>
          <img src="/arrow.png" alt="" />
        </button>
      </form> */}
    </>
  );
};

export default NewPrompt;
