import { useState, useEffect, useRef } from "react";

// ─── Palettes ────────────────────────────────────────────────────
const LIGHT = {
  primary:"#1D6EEB", mint:"#00B894", danger:"#E05260", amber:"#F59E0B",
  ink:"#1A1A2E", muted:"#6B7280", subtle:"#9CA3AF",
  border:"#EDE9F6", bgSoft:"#FAF8FF", bg:"#FFFFFF",
  navBg:"rgba(255,255,255,.97)", purpleTint:"#F3EEFF",
  mintTint:"#E6FAF5", redTint:"#FDECEA", amberTint:"#FFF8EC",
  cardBg:"#FFFFFF", cardBorder:"#EDE9F6",
};
const DARK = {
  primary:"#A78BFA", mint:"#34D399", danger:"#F87171", amber:"#FBBF24",
  ink:"#F0EEFF", muted:"#9CA3AF", subtle:"#6B7280",
  border:"#1E1B4B", bgSoft:"#0D0B1A", bg:"#07050F",
  navBg:"rgba(7,5,15,.97)", purpleTint:"#13103A",
  mintTint:"#052E20", redTint:"#1F0A0A", amberTint:"#1C1000",
  cardBg:"#100D24", cardBorder:"#1E1B4B",
};

const MC_L = ["#00B894","#1D6EEB","#F59E0B","#E05260"];
const MC_D = ["#34D399","#A78BFA","#FBBF24","#F87171"];

const BASE_URL = process.env.NEXT_PUBLIC_API_URL;

// ─── API ─────────────────────────────────────────────────────────
const callAPI = async (path, body) => {
  const r = await fetch(`${BASE_URL}${path}`, {
    method:"POST", headers:{"Content-Type":"application/json"},
    body: JSON.stringify(body),
  });
  return r.json();
};

const pingServer = async () => {
  try {
    const r = await fetch(`${BASE_URL}/docs`, { method:"GET", signal: AbortSignal.timeout(3000) });
    return r.ok || r.status < 500;
  } catch { return false; }
};

const mapPredict = (q, r) => ({
  topic:q, score:Math.round(r.confidence*100),
  sentiment:r.prediction,
  positive:Math.round(r.positive*100),
  neutral: Math.round(r.neutral*100),
  negative:Math.round(r.negative*100),
  breakdown:[
    {label:"Positive",value:Math.round(r.positive*100),color:"#00B894"},
    {label:"Neutral", value:Math.round(r.neutral*100), color:"#F59E0B"},
    {label:"Negative",value:Math.round(r.negative*100),color:"#E05260"},
  ],
  insights:[
    {icon:"🎯",tag:"Model",    tc:"#6C3FC5",tb:"#F3EEFF",text:`RoBERTa detected ${r.prediction} with ${Math.round(r.confidence*100)}% confidence`},
    {icon:"📊",tag:"Breakdown",tc:"#00B894",tb:"#E6FAF5",text:`Positive ${Math.round(r.positive*100)}% · Neutral ${Math.round(r.neutral*100)}% · Negative ${Math.round(r.negative*100)}%`},
    {icon:"🍽️",tag:"Dataset",  tc:"#B7770D",tb:"#FFF8EC",text:"Zomato Reviews — Food & Dining Sentiment Corpus"},
    {icon:"⚡",tag:"Engine",   tc:"#E05260",tb:"#FDECEA",text:"RoBERTa Transformer · Fine-tuned on 2900K+ restaurant reviews"},
  ],
});

const mapCompare = (q, json) => {
  const sorted = [...json.comparison].sort((a,b) => b.confidence - a.confidence);
  const best   = sorted[0];
  return {
    query:q, score:Math.round(best.confidence*100),
    sentiment:best.prediction, models:sorted,
    insights:[
      {icon:"🏆",tag:"Best",     tc:"#00B894",tb:"#E6FAF5",text:`${best.model.toUpperCase()} leads — ${Math.round(best.confidence*100)}% confidence → ${best.prediction}`},
      {icon:"🤖",tag:"Consensus",tc:"#6C3FC5",tb:"#F3EEFF",text:`${sorted.filter(m=>m.prediction===best.prediction).length}/${sorted.length} models agree on "${best.prediction}"`},
      {icon:"📐",tag:"Spread",   tc:"#B7770D",tb:"#FFF8EC",text:`Range: ${Math.round(sorted[sorted.length-1].confidence*100)}%–${Math.round(best.confidence*100)}% across all models`},
    ],
  };
};

// ─── PDF Export ───────────────────────────────────────────────────
const exportPDF = (data, dataB, query) => {
  if (!data) return;
  const now=new Date();
  const ds=now.toLocaleDateString("en-IN",{year:"numeric",month:"long",day:"numeric"});
  const ts=now.toLocaleTimeString("en-IN",{hour:"2-digit",minute:"2-digit"});
  const id=`ML-${Date.now().toString(36).toUpperCase()}`;
  const isC=!!dataB;
  const SG={Positive:["#00B894","#047857"],Negative:["#E05260","#9B1C1C"],Neutral:["#F59E0B","#92400E"]};
  const SE={Positive:"😊",Negative:"😞",Neutral:"😐"};
  const sg=SG[data.sentiment]||SG.Neutral;
  const emo=SE[data.sentiment]||"🔍";
  const MC2=MC_L;

  const ring=(val,color,sz=80)=>{
    const r=sz/2-9,c=(2*Math.PI*r).toFixed(1),off=(c*(1-val/100)).toFixed(1),cx=sz/2,cy=sz/2;
    return `<svg width="${sz}" height="${sz}" viewBox="0 0 ${sz} ${sz}"><circle cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="#F3EEFF" stroke-width="8"/><circle cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="${color}" stroke-width="8" stroke-dasharray="${c}" stroke-dashoffset="${off}" stroke-linecap="round" transform="rotate(-90 ${cx} ${cy})"/><text x="${cx}" y="${cy}" text-anchor="middle" dominant-baseline="central" font-family="Segoe UI" font-size="${sz>72?14:11}" font-weight="800" fill="#1A1A2E">${val}%</text></svg>`;
  };
  const hbar=(label,val,color,note="")=>`
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:9px">
      <span style="width:108px;font-size:12px;color:#6B7280;font-weight:500;flex-shrink:0">${label}</span>
      <div style="flex:1;height:8px;background:#F3EEFF;border-radius:99px;overflow:hidden">
        <div style="width:${val}%;height:100%;background:${color};border-radius:99px"></div>
      </div>
      ${note?`<span style="font-size:10px;color:#9CA3AF;margin-right:4px;flex-shrink:0">${note}</span>`:""}
      <span style="font-size:12px;font-weight:800;color:${color};width:34px;text-align:right;flex-shrink:0">${val}%</span>
    </div>`;
  const kpi=(icon,label,val,color,bg)=>`
    <div style="background:${bg};border:1.5px solid ${color}35;border-radius:13px;padding:16px 12px;text-align:center">
      <div style="font-size:22px;margin-bottom:5px">${icon}</div>
      <div style="font-size:28px;font-weight:900;color:${color};font-family:Segoe UI;line-height:1">${val}%</div>
      <div style="font-size:10px;color:#6B7280;font-weight:600;text-transform:uppercase;letter-spacing:1.2px;margin-top:4px">${label}</div>
    </div>`;
  const irow=(i)=>`
    <div style="display:flex;align-items:flex-start;gap:11px;padding:10px 13px;background:#FAF8FF;border:1px solid ${i.tc}22;border-radius:9px;margin-bottom:7px">
      <span style="font-size:16px;flex-shrink:0">${i.icon}</span>
      <span style="font-size:12px;color:#374151;flex:1;line-height:1.6">${i.text}</span>
      <span style="font-size:9px;font-weight:700;padding:2px 9px;border-radius:999px;background:${i.tb};color:${i.tc};white-space:nowrap">${i.tag}</span>
    </div>`;

  let body = "";
  if (!isC) {
    body = `
    <div style="background:linear-gradient(135deg,${sg[0]} 0%,${sg[1]} 100%);border-radius:18px;padding:28px 32px;color:#fff;display:flex;align-items:center;justify-content:space-between;margin-bottom:22px;position:relative;overflow:hidden">
      <div style="position:absolute;top:-55px;right:-55px;width:220px;height:220px;border-radius:50%;background:rgba(255,255,255,.09)"></div>
      <div style="position:absolute;bottom:-70px;left:270px;width:240px;height:240px;border-radius:50%;background:rgba(255,255,255,.06)"></div>
      <div style="position:relative;z-index:1">
        <div style="font-size:9px;letter-spacing:2.5px;text-transform:uppercase;opacity:.7;margin-bottom:8px">Sentiment Analysis · RoBERTa v1 · Zomato Dataset</div>
        <div style="font-size:14px;opacity:.85;max-width:360px;line-height:1.6;margin-bottom:16px;font-style:italic;padding:8px 14px;background:rgba(255,255,255,.13);border-radius:8px;border-left:3px solid rgba(255,255,255,.55)">"${data.topic}"</div>
        <div style="display:flex;align-items:center;gap:14px">
          <span style="font-size:44px">${emo}</span>
          <div>
            <div style="font-size:10px;letter-spacing:3px;text-transform:uppercase;opacity:.7;margin-bottom:4px">Detected as</div>
            <div style="font-size:40px;font-weight:900;letter-spacing:-2px;line-height:.92;font-family:Segoe UI">${data.sentiment.toUpperCase()}</div>
          </div>
        </div>
      </div>
      <div style="position:relative;z-index:1;text-align:center;flex-shrink:0">
        <div style="font-size:9px;letter-spacing:2px;text-transform:uppercase;opacity:.7;margin-bottom:8px">Confidence</div>
        ${ring(data.score,"rgba(255,255,255,.92)",98)}
        <div style="font-size:10px;opacity:.7;margin-top:5px">RoBERTa v1</div>
      </div>
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-bottom:20px">
      ${kpi("✅","Positive",data.positive,"#00B894","#E6FAF5")}
      ${kpi("🔘","Neutral", data.neutral, "#F59E0B","#FFF8EC")}
      ${kpi("❌","Negative",data.negative,"#E05260","#FDECEA")}
    </div>
    <div style="display:grid;grid-template-columns:1.8fr 1fr;gap:13px;margin-bottom:20px">
      <div style="background:#FAF8FF;border:1.5px solid #EDE9F6;border-radius:13px;padding:18px">
        <div style="font-size:9px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:#9CA3AF;margin-bottom:13px">Confidence Breakdown</div>
        ${hbar("Positive",data.positive,"#00B894")}${hbar("Neutral",data.neutral,"#F59E0B")}${hbar("Negative",data.negative,"#E05260")}
      </div>
      <div style="display:flex;flex-direction:column;gap:10px">
        <div style="flex:1;background:#F3EEFF;border:1.5px solid #6C3FC522;border-radius:12px;padding:14px">
          <div style="font-size:9px;text-transform:uppercase;color:#9CA3AF;letter-spacing:1.5px;margin-bottom:5px">Default Model</div>
          <div style="font-size:14px;font-weight:800;color:#6C3FC5">RoBERTa v1</div>
          <div style="font-size:11px;color:#6B7280;margin-top:3px">Transformer · Fine-tuned</div>
        </div>
        <div style="flex:1;background:#E6FAF5;border:1.5px solid #00B89422;border-radius:12px;padding:14px">
          <div style="font-size:9px;text-transform:uppercase;color:#9CA3AF;letter-spacing:1.5px;margin-bottom:5px">Dataset</div>
          <div style="font-size:14px;font-weight:800;color:#00B894">Zomato Reviews</div>
          <div style="font-size:11px;color:#6B7280;margin-top:3px">Food & Dining Corpus</div>
        </div>
      </div>
    </div>
    <div style="background:#fff;border:1.5px solid #EDE9F6;border-radius:13px;padding:18px">
      <div style="font-size:9px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:#9CA3AF;margin-bottom:12px">Smart Insights</div>
      ${data.insights.map(irow).join("")}
    </div>`;
  } else {
    body = `
    <div style="background:linear-gradient(135deg,#1A1A2E 0%,#3B1F7A 55%,#00B894 100%);border-radius:18px;padding:26px 30px;color:#fff;display:flex;align-items:center;justify-content:space-between;margin-bottom:18px;position:relative;overflow:hidden">
      <div style="position:absolute;top:-50px;right:-50px;width:200px;height:200px;border-radius:50%;background:rgba(255,255,255,.07)"></div>
      <div style="position:relative;z-index:1">
        <div style="font-size:9px;letter-spacing:2.5px;text-transform:uppercase;opacity:.65;margin-bottom:8px">Multi-Model Comparison · Zomato Dataset</div>
        <div style="font-size:19px;font-weight:800;margin-bottom:5px">"${query}"</div>
        <div style="font-size:11px;opacity:.65;margin-bottom:13px">RoBERTa · DistilRoBERTa · BERT · ALBERT</div>
        <div style="display:inline-flex;align-items:center;gap:8px;background:rgba(255,255,255,.15);border-radius:999px;padding:5px 15px">
          <span style="font-size:15px">${emo}</span><span style="font-size:12px;font-weight:700">Consensus: ${data.sentiment}</span>
        </div>
      </div>
      <div style="position:relative;z-index:1;text-align:center;flex-shrink:0">
        <div style="font-size:9px;letter-spacing:2px;text-transform:uppercase;opacity:.65;margin-bottom:7px">Best Model</div>
        ${ring(dataB.score,"rgba(255,255,255,.92)",90)}
        <div style="font-size:11px;opacity:.8;margin-top:5px;font-weight:700">${dataB.models[0]?.model.toUpperCase()}</div>
      </div>
    </div>
    <div style="background:#FAF8FF;border:1.5px solid #6C3FC522;border-radius:13px;padding:18px;margin-bottom:15px">
      <div style="font-size:9px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:#6C3FC5;margin-bottom:12px">RoBERTa — Default Model Result</div>
      <div style="display:flex;align-items:center;gap:18px">
        ${ring(data.score,"#6C3FC5",68)}
        <div style="flex:1">
          <div style="font-size:16px;font-weight:800;color:#1A1A2E;margin-bottom:11px">${data.sentiment} <span style="font-size:12px;color:#9CA3AF;font-weight:400">· ${data.score}% confidence</span></div>
          ${hbar("Positive",data.positive,"#00B894")}${hbar("Neutral",data.neutral,"#F59E0B")}${hbar("Negative",data.negative,"#E05260")}
        </div>
      </div>
    </div>
    <div style="background:#fff;border:1.5px solid #EDE9F6;border-radius:13px;padding:18px;margin-bottom:15px">
      <div style="font-size:9px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:#9CA3AF;margin-bottom:15px">All Models — Confidence Rings</div>
      <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;text-align:center">
        ${dataB.models.map((m,i)=>{const col=MC2[i],iB=i===0;return`<div style="background:${iB?`linear-gradient(160deg,${col}1E,${col}08)`:"#FAF8FF"};border:${iB?`2px solid ${col}`:"1.5px solid #EDE9F6"};border-radius:12px;padding:14px 8px;position:relative">${iB?`<div style="position:absolute;top:-11px;left:50%;transform:translateX(-50%);background:${col};color:#fff;font-size:9px;font-weight:700;padding:2px 10px;border-radius:999px;letter-spacing:1px;white-space:nowrap;box-shadow:0 2px 8px ${col}55">⭐ BEST</div>`:""} ${ring(Math.round(m.confidence*100),col,70)}<div style="font-size:10px;font-weight:800;color:#1A1A2E;margin-top:8px">${m.model.toUpperCase()}</div><div style="margin-top:5px"><span style="font-size:9px;font-weight:700;padding:2px 8px;border-radius:999px;background:${col}22;color:${col}">${m.prediction}</span></div></div>`;}).join("")}
      </div>
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:13px">
      <div style="background:#FAF8FF;border:1.5px solid #EDE9F6;border-radius:13px;padding:17px">
        <div style="font-size:9px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:#9CA3AF;margin-bottom:12px">Model Confidence Comparison</div>
        ${dataB.models.map((m,i)=>hbar(m.model.toUpperCase(),Math.round(m.confidence*100),MC2[i],m.prediction)).join("")}
      </div>
      <div style="background:#fff;border:1.5px solid #EDE9F6;border-radius:13px;padding:17px">
        <div style="font-size:9px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:#9CA3AF;margin-bottom:12px">Multi-Model Insights</div>
        ${dataB.insights.map(irow).join("")}
      </div>
    </div>`;
  }

  const html = `<!DOCTYPE html><html><head><meta charset="UTF-8"/><title>MoodLens · ${id}</title>
  <style>*{box-sizing:border-box;margin:0;padding:0;}body{font-family:'Segoe UI',sans-serif;background:#F5F0FF;color:#1A1A2E;}.page{max-width:800px;margin:0 auto;padding:34px 30px;}@media print{body{background:#fff;}.page{padding:14px;}@page{margin:8mm 10mm;size:A4;}}</style>
  </head><body><div class="page">
  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:24px;padding-bottom:16px;border-bottom:3px solid #6C3FC5">
    <div style="display:flex;align-items:center;gap:13px">
      <div style="width:48px;height:48px;background:linear-gradient(135deg,#6C3FC5,#00B894);border-radius:14px;display:flex;align-items:center;justify-content:center;font-size:26px">📊</div>
      <div>
        <div style="font-size:26px;font-weight:900;letter-spacing:-1px"><span style="color:#6C3FC5">Mo</span><span style="color:#00B894">od</span><span style="color:#6C3FC5">Le</span><span style="color:#00B894">ns</span></div>
        <div style="font-size:10px;color:#9CA3AF;margin-top:2px">Sentiment Intelligence Platform · NLP Engine</div>
      </div>
    </div>
    <div style="text-align:right">
      <div style="font-size:12px;font-weight:700;color:#6C3FC5;margin-bottom:3px">${isC?"Multi-Model Comparison Report":"Sentiment Analysis Report"}</div>
      <div style="font-size:11px;color:#9CA3AF">${ds} · ${ts}</div>
      <div style="display:inline-block;margin-top:4px;padding:2px 10px;background:#F3EEFF;border-radius:999px;font-size:9px;color:#6C3FC5;font-weight:700">ID: ${id}</div>
    </div>
  </div>
  ${body}
  <div style="margin-top:26px;padding-top:13px;border-top:1.5px solid #EDE9F6;display:flex;justify-content:space-between;align-items:center">
    <div style="font-size:9px;color:#C4B5FD;letter-spacing:.8px;text-transform:uppercase">MoodLens AI · Confidential · Analytical Use Only</div>
    <div style="font-size:9px;color:#9CA3AF">v1.0 · Zomato Dataset · ${ds}</div>
  </div>
  </div></body></html>`;

  const blob=new Blob([html],{type:"text/html"});
  const url=URL.createObjectURL(blob);
  const win=window.open(url,"_blank");
  if (win) win.addEventListener("load",()=>setTimeout(()=>win.print(),600));
};

// ─── Animated Counter ─────────────────────────────────────────────
function Counter({to}){
  const [n,setN]=useState(0);
  useEffect(()=>{
    let raf,s;
    const tick=ts=>{if(!s)s=ts;const p=Math.min((ts-s)/1000,1);setN(Math.floor((1-Math.pow(1-p,4))*to));if(p<1)raf=requestAnimationFrame(tick);};
    raf=requestAnimationFrame(tick);
    return()=>cancelAnimationFrame(raf);
  },[to]);
  return <>{n}</>;
}

// ─── SVG Ring ─────────────────────────────────────────────────────
function Ring({val,color,size=84,thick=8,bgColor}){
  const r=size/2-thick, c=2*Math.PI*r;
  const [d,setD]=useState(c);
  useEffect(()=>{const t=setTimeout(()=>setD(c*(1-val/100)),180);return()=>clearTimeout(t);},[val,c]);
  return(
    <svg width={size} height={size} style={{transform:"rotate(-90deg)",flexShrink:0}}>
      <circle cx={size/2} cy={size/2} r={r} fill="none" stroke={bgColor||"rgba(108,63,197,.1)"} strokeWidth={thick}/>
      <circle cx={size/2} cy={size/2} r={r} fill="none" stroke={color} strokeWidth={thick}
        strokeDasharray={c} strokeDashoffset={d} strokeLinecap="round"
        style={{transition:"stroke-dashoffset 1.4s cubic-bezier(.34,1.1,.64,1)"}}/>
    </svg>
  );
}

// ─── Horizontal Bar ───────────────────────────────────────────────
function Bar({label,value,color,delay,note,P}){
  const [w,setW]=useState(0);
  useEffect(()=>{const t=setTimeout(()=>setW(value),delay);return()=>clearTimeout(t);},[value,delay]);
  return(
    <div style={{display:"flex",alignItems:"center",gap:12,marginBottom:11}}>
      <span style={{width:118,fontSize:12.5,color:P.muted,flexShrink:0}}>{label}</span>
      <div style={{flex:1,height:8,background:P.bgSoft,borderRadius:99,overflow:"hidden"}}>
        <div style={{width:`${w}%`,height:"100%",background:color,borderRadius:99,transition:`width 1s cubic-bezier(.34,1.1,.64,1) ${delay}ms`}}/>
      </div>
      {note&&<span style={{fontSize:11,color:P.subtle,flexShrink:0,marginRight:4}}>{note}</span>}
      <span style={{width:30,fontSize:12,fontWeight:700,color,textAlign:"right",flexShrink:0}}>{value}</span>
    </div>
  );
}

// ─── Queue Loader ─────────────────────────────────────────────────
function QueueLoader({query,P,TOTAL}){
  const [elapsed,setElapsed]=useState(0);
  const [step,setStep]=useState(0);
  const steps=[
    {label:"Initialising NLP engine",      icon:"🔧"},
    {label:"Pre-processing your text",     icon:"📝"},
    {label:"Running RoBERTa inference",    icon:"⚡"},
    {label:"Computing sentiment scores",   icon:"📊"},
    {label:"Finalising results",           icon:"✅"},
  ];
  useEffect(()=>{
    if(elapsed>=TOTAL) return;
    const t=setInterval(()=>setElapsed(e=>parseFloat(Math.min(e+0.05,TOTAL).toFixed(2))),50);
    return()=>clearInterval(t);
  },[elapsed,TOTAL]);
  useEffect(()=>{setStep(Math.min(Math.floor((elapsed/TOTAL)*steps.length), steps.length-1));},[elapsed, TOTAL, steps.length]);
  const pct=Math.min((elapsed/TOTAL)*100,100);
  const r=48, circ=2*Math.PI*r;
  const offset=circ*(1-pct/100);
  return(
    <div style={{display:"flex",flexDirection:"column",alignItems:"center",justifyContent:"center",minHeight:"68vh",padding:"32px 24px"}}>
      <div style={{width:"100%",maxWidth:500,background:P.cardBg,border:`1.5px solid ${P.border}`,borderRadius:22,padding:"38px 36px",boxShadow:`0 20px 60px rgba(108,63,197,.12)`,position:"relative",overflow:"hidden"}}>
        <div style={{position:"absolute",top:-40,right:-40,width:160,height:160,borderRadius:"50%",background:`radial-gradient(circle,${P.purpleTint},transparent)`,opacity:.7,pointerEvents:"none"}}/>
        <div style={{position:"absolute",bottom:-40,left:-30,width:140,height:140,borderRadius:"50%",background:`radial-gradient(circle,${P.mintTint},transparent)`,opacity:.7,pointerEvents:"none"}}/>
        <div style={{position:"relative",zIndex:1}}>
          <div style={{display:"flex",justifyContent:"center",marginBottom:26}}>
            <div style={{position:"relative",width:114,height:114}}>
              <svg width="114" height="114" style={{position:"absolute",top:0,left:0,transform:"rotate(-90deg)"}}>
                <circle cx="57" cy="57" r={r} fill="none" stroke={P.border} strokeWidth="9"/>
                <circle cx="57" cy="57" r={r} fill="none" stroke={P.primary} strokeWidth="9"
                  strokeDasharray={circ} strokeDashoffset={offset} strokeLinecap="round"
                  style={{transition:"stroke-dashoffset .08s linear"}}/>
              </svg>
              <svg width="114" height="114" style={{position:"absolute",top:0,left:0,animation:"qspin 1.8s linear infinite"}}>
                <circle cx="57" cy="57" r="34" fill="none" stroke={P.mint} strokeWidth="3"
                  strokeDasharray="50 165" strokeLinecap="round"/>
              </svg>
              <div style={{position:"absolute",inset:0,display:"flex",flexDirection:"column",alignItems:"center",justifyContent:"center"}}>
                <span style={{fontSize:26,fontWeight:900,fontFamily:"'Nunito',sans-serif",color:P.primary,lineHeight:1}}>{Math.round(pct)}</span>
                <span style={{fontSize:10,color:P.subtle,marginTop:1,letterSpacing:"1px"}}>%</span>
              </div>
            </div>
          </div>
          <div style={{textAlign:"center",marginBottom:22}}>
            <div style={{fontSize:10,letterSpacing:"2.5px",textTransform:"uppercase",color:P.subtle,marginBottom:6}}>Please Wait · Your message is in queue</div>
            <div style={{fontSize:18,fontWeight:800,color:P.ink,fontFamily:"'Nunito',sans-serif",letterSpacing:"-.4px",marginBottom:8}}>
              {steps[step]?.icon} {steps[step]?.label}…
            </div>
            <div style={{fontSize:12,color:P.subtle,fontStyle:"italic",padding:"6px 16px",background:P.bgSoft,borderRadius:8,border:`1px solid ${P.border}`,display:"inline-block",maxWidth:360,overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}}>
              "{query.length>52?query.slice(0,52)+"…":query}"
            </div>
          </div>
          <div style={{display:"flex",flexDirection:"column",gap:5}}>
            {steps.map((s,i)=>{
              const done=i<step, active=i===step;
              return(
                <div key={i} style={{display:"flex",alignItems:"center",gap:10,padding:"8px 12px",borderRadius:10,background:active?P.purpleTint:"transparent",border:active?`1px solid ${P.primary}33`:"1px solid transparent",transition:"all .25s"}}>
                  <div style={{width:26,height:26,borderRadius:"50%",flexShrink:0,display:"flex",alignItems:"center",justifyContent:"center",fontSize:done?11:12,fontWeight:700,color:"#fff",background:done?P.mint:active?P.primary:P.border,transition:"background .3s",boxShadow:active?`0 0 10px ${P.primary}44`:"none"}}>
                    {done?"✓":s.icon}
                  </div>
                  <span style={{fontSize:12,fontWeight:active?600:400,color:active?P.ink:done?P.muted:P.subtle,transition:"all .25s"}}>{s.label}</span>
                  {done&&<div style={{marginLeft:"auto",width:6,height:6,borderRadius:"50%",background:P.mint}}/>}
                </div>
              );
            })}
          </div>
          <div style={{marginTop:18,display:"flex",justifyContent:"space-between"}}>
            <span style={{fontSize:11,color:P.subtle}}>Est. {Math.max(0,Math.ceil(TOTAL-elapsed))}s remaining</span>
            <span style={{fontSize:11,color:P.subtle,letterSpacing:".8px",textTransform:"uppercase"}}>RoBERTa · Zomato</span>
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── Prediction Hero ──────────────────────────────────────────────
function PredictionHero({sentiment,score,topic,P}){
  const cfg={
    Positive:{g:"linear-gradient(135deg,#00B894 0%,#059669 50%,#047857 100%)",glow:"rgba(0,184,148,.3)",emoji:"😊",lbl:"POSITIVE"},
    Negative:{g:"linear-gradient(135deg,#E05260 0%,#C0392B 50%,#7F1D1D 100%)",glow:"rgba(224,82,96,.3)",emoji:"😞",lbl:"NEGATIVE"},
    Neutral: {g:"linear-gradient(135deg,#F59E0B 0%,#D97706 50%,#92400E 100%)",glow:"rgba(245,158,11,.3)",emoji:"😐",lbl:"NEUTRAL"},
  }[sentiment]||{g:"linear-gradient(135deg,#6C3FC5,#3B0764)",glow:"rgba(108,63,197,.3)",emoji:"🔍",lbl:"UNKNOWN"};
  const [vis,setVis]=useState(false);
  useEffect(()=>{const t=setTimeout(()=>setVis(true),80);return()=>clearTimeout(t);},[]);
  const r=46, c=2*Math.PI*r;
  const [d,setD]=useState(c);
  useEffect(()=>{const t=setTimeout(()=>setD(c*(1-score/100)),260);return()=>clearTimeout(t);},[score,c]);
  return(
    <div style={{borderRadius:20,overflow:"hidden",marginBottom:16,boxShadow:`0 20px 60px ${cfg.glow},0 4px 16px rgba(0,0,0,.1)`,opacity:vis?1:0,transform:vis?"translateY(0)":"translateY(18px)",transition:"opacity .55s ease,transform .55s ease"}}>
      <div style={{background:cfg.g,padding:"30px 36px",color:"#fff",display:"flex",alignItems:"center",justifyContent:"space-between",position:"relative",overflow:"hidden"}}>
        <div style={{position:"absolute",top:-60,right:-60,width:240,height:240,borderRadius:"50%",background:"rgba(255,255,255,.08)",pointerEvents:"none"}}/>
        <div style={{position:"absolute",bottom:-80,right:220,width:260,height:260,borderRadius:"50%",background:"rgba(255,255,255,.05)",pointerEvents:"none"}}/>
        <div style={{position:"absolute",top:20,left:-40,width:130,height:130,borderRadius:"50%",background:"rgba(255,255,255,.05)",pointerEvents:"none"}}/>
        <div style={{position:"relative",zIndex:2}}>
          <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:10}}>
            <div style={{width:7,height:7,borderRadius:"50%",background:"rgba(255,255,255,.75)",animation:"pulse 2s infinite"}}/>
            <span style={{fontSize:10,letterSpacing:"2.5px",textTransform:"uppercase",opacity:.75}}>Live Sentiment · RoBERTa v1 · Zomato Dataset</span>
          </div>
          <div style={{fontSize:"clamp(12px,1.5vw,14px)",opacity:.82,maxWidth:380,lineHeight:1.6,marginBottom:18,fontStyle:"italic",padding:"9px 14px",background:"rgba(255,255,255,.12)",borderRadius:9,borderLeft:"4px solid rgba(255,255,255,.5)"}}>"{topic}"</div>
          <div style={{display:"flex",alignItems:"center",gap:16}}>
            <span style={{fontSize:"clamp(42px,6vw,56px)"}}>{cfg.emoji}</span>
            <div>
              <div style={{fontSize:10,letterSpacing:"3.5px",textTransform:"uppercase",opacity:.72,marginBottom:5}}>Detected as</div>
              <div style={{fontSize:"clamp(32px,5vw,52px)",fontWeight:900,letterSpacing:"-2px",lineHeight:.9,fontFamily:"'Nunito',sans-serif",textShadow:"0 3px 16px rgba(0,0,0,.2)"}}>{cfg.lbl}</div>
            </div>
          </div>
        </div>
        <div style={{position:"relative",zIndex:2,textAlign:"center",flexShrink:0}}>
          <div style={{fontSize:10,letterSpacing:"2px",textTransform:"uppercase",opacity:.72,marginBottom:9}}>Confidence</div>
          <div style={{position:"relative",width:124,height:124}}>
            <div style={{position:"absolute",inset:-6,borderRadius:"50%",background:"radial-gradient(circle,rgba(255,255,255,.15) 0%,transparent 70%)"}}/>
            <svg width="124" height="124" style={{position:"absolute",top:0,left:0}}>
              <circle cx="62" cy="62" r={r} fill="none" stroke="rgba(255,255,255,.2)" strokeWidth="10"/>
              <circle cx="62" cy="62" r={r} fill="none" stroke="rgba(255,255,255,.92)" strokeWidth="10"
                strokeDasharray={c} strokeDashoffset={d} strokeLinecap="round" transform="rotate(-90 62 62)"
                style={{transition:"stroke-dashoffset 1.5s cubic-bezier(.34,1.1,.64,1)"}}/>
            </svg>
            <div style={{position:"absolute",inset:0,display:"flex",flexDirection:"column",alignItems:"center",justifyContent:"center"}}>
              <span style={{fontSize:30,fontWeight:900,fontFamily:"'Nunito',sans-serif",lineHeight:1}}><Counter to={score}/></span>
              <span style={{fontSize:11,opacity:.72,marginTop:2}}>/ 100</span>
            </div>
          </div>
        </div>
      </div>
      <div style={{background:"rgba(0,0,0,.28)",backdropFilter:"blur(10px)",display:"grid",gridTemplateColumns:"1fr 1fr 1fr"}}>
        {[{l:"Model",v:"RoBERTa v1"},{l:"Dataset",v:"Zomato Reviews"},{l:"Status",v:"✅ Complete"}].map((item,i)=>(
          <div key={i} style={{padding:"11px 18px",borderRight:i<2?"1px solid rgba(255,255,255,.1)":undefined,textAlign:"center"}}>
            <div style={{fontSize:9,letterSpacing:"1.5px",textTransform:"uppercase",color:"rgba(255,255,255,.55)",marginBottom:2}}>{item.l}</div>
            <div style={{fontSize:12,fontWeight:700,color:"rgba(255,255,255,.9)"}}>{item.v}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── Model Ring Card ──────────────────────────────────────────────
function ModelCard({model,confidence,prediction,color,isBest,delay,P}){
  const val=Math.round(confidence*100);
  const [vis,setVis]=useState(false);
  useEffect(()=>{const t=setTimeout(()=>setVis(true),delay);return()=>clearTimeout(t);},[delay]);
  return(
    <div style={{position:"relative",background:isBest?`linear-gradient(160deg,${color}1E 0%,${color}08 100%)`:P.cardBg,border:isBest?`2px solid ${color}`:`1.5px solid ${P.cardBorder}`,borderRadius:16,padding:"22px 14px",display:"flex",flexDirection:"column",alignItems:"center",gap:10,boxShadow:isBest?`0 8px 28px ${color}30,0 2px 8px rgba(0,0,0,.06)`:"0 2px 8px rgba(108,63,197,.05)",opacity:vis?1:0,transform:vis?"translateY(0)":"translateY(20px)",transition:"opacity .45s ease,transform .45s ease"}}>
      {isBest&&<div style={{position:"absolute",top:-13,left:"50%",transform:"translateX(-50%)",background:`linear-gradient(90deg,${color},${color}CC)`,color:"#fff",fontSize:10,fontWeight:700,padding:"3px 14px",borderRadius:999,letterSpacing:"1px",textTransform:"uppercase",whiteSpace:"nowrap",boxShadow:`0 3px 10px ${color}55`}}>⭐ Best Model</div>}
      <div style={{position:"relative",width:90,height:90}}>
        <Ring val={val} color={color} size={90} thick={9} bgColor={P.border}/>
        <div style={{position:"absolute",inset:0,display:"flex",flexDirection:"column",alignItems:"center",justifyContent:"center"}}>
          <span style={{fontSize:20,fontWeight:900,color:P.ink,fontFamily:"'Nunito',sans-serif",lineHeight:1}}><Counter to={val}/></span>
          <span style={{fontSize:10,color:P.subtle,marginTop:1}}>%</span>
        </div>
      </div>
      <div style={{textAlign:"center"}}>
        <div style={{fontSize:12,fontWeight:800,color:P.ink,marginBottom:5}}>{model.toUpperCase()}</div>
        <span style={{fontSize:11,fontWeight:700,padding:"3px 11px",borderRadius:999,background:`${color}18`,color,border:`1px solid ${color}44`}}>{prediction}</span>
        <div style={{fontSize:11,color:P.subtle,marginTop:6,fontWeight:600}}>{val}% confidence</div>
      </div>
    </div>
  );
}

// ─── Server Status ────────────────────────────────────────────────
function ServerStatus({P}){
  const [status,setStatus]=useState("checking");
  const check=async()=>{setStatus("checking");const ok=await pingServer();setStatus(ok?"online":"offline");};
  useEffect(()=>{check();const iv=setInterval(check,15000);return()=>clearInterval(iv);},[]);
  const cfg={
    checking:{color:"#F59E0B",bg:"#FFF8EC",border:"#F59E0B44",dot:"#F59E0B",label:"Checking…",pulse:true},
    online:  {color:"#00B894",bg:"#E6FAF5",border:"#00B89444",dot:"#00B894",label:"Server Online",pulse:false},
    offline: {color:"#E05260",bg:"#FDECEA",border:"#E0526044",dot:"#E05260",label:"Server Offline",pulse:false},
  }[status];
  return(
    <div onClick={check} title={status==="offline"?"Click to retry":"Click to refresh"}
      style={{display:"flex",alignItems:"center",gap:7,padding:"5px 12px",borderRadius:999,background:cfg.bg,border:`1.5px solid ${cfg.border}`,cursor:"pointer",transition:"all .2s",flexShrink:0}}>
      <div style={{position:"relative",width:8,height:8,flexShrink:0}}>
        <div style={{width:8,height:8,borderRadius:"50%",background:cfg.dot}}/>
        {cfg.pulse&&<div style={{position:"absolute",inset:-2,borderRadius:"50%",background:cfg.dot,opacity:.4,animation:"pingAnim 1.2s ease-out infinite"}}/>}
        {status==="online"&&<div style={{position:"absolute",inset:-2,borderRadius:"50%",background:cfg.dot,opacity:.25,animation:"pingAnim 2s ease-out infinite"}}/>}
      </div>
      <span style={{fontSize:11,fontWeight:700,color:cfg.color,whiteSpace:"nowrap"}}>
        {status==="offline"?<><span style={{marginRight:4}}>⚠️</span>{cfg.label}</>:cfg.label}
      </span>
      {status==="offline"&&<span style={{fontSize:10,color:cfg.color,opacity:.7}}>↻</span>}
    </div>
  );
}

// ─── Live Clock ───────────────────────────────────────────────────
function LiveClock({P}){
  const [t,setT]=useState(new Date());
  useEffect(()=>{const i=setInterval(()=>setT(new Date()),1000);return()=>clearInterval(i);},[]);
  const pad=n=>n.toString().padStart(2,"0");
  return(
    <div style={{display:"flex",flexDirection:"column",alignItems:"flex-end",gap:1,flexShrink:0}}>
      <div style={{fontFamily:"'Nunito',sans-serif",fontSize:15,fontWeight:900,letterSpacing:"1px",color:P.primary}}>
        {pad(t.getHours())}<span style={{opacity:.4,animation:"blink 1s step-end infinite"}}>:</span>{pad(t.getMinutes())}<span style={{opacity:.4,animation:"blink 1s step-end infinite"}}>:</span><span style={{fontSize:12,opacity:.65}}>{pad(t.getSeconds())}</span>
      </div>
      <div style={{fontSize:9,color:P.subtle,letterSpacing:".3px"}}>{t.toLocaleDateString("en-IN",{day:"2-digit",month:"short",year:"numeric"})}</div>
    </div>
  );
}

// ─── Logo ─────────────────────────────────────────────────────────
// Dark mode  → Yellow (#F59E0B) + White (#FFFFFF)
// Light mode → Uber Blue (#1D6EEB) + Black (#000000)
function Logo({size=52, dark=false}){
  const c1 = dark ? "#F59E0B" : "#1D6EEB";
  const c2 = dark ? "#FFFFFF" : "#000000";
  const L=[
    {ch:"M",c:c1},{ch:"o",c:c2},
    {ch:"o",c:c1},{ch:"d",c:c2},
    {ch:"L",c:c1},{ch:"e",c:c2},
    {ch:"n",c:c1},{ch:"s",c:c2},
  ];
  return(
    <span style={{fontFamily:"'Nunito',sans-serif",fontWeight:900,fontSize:size,letterSpacing:"-1px",lineHeight:1,userSelect:"none",cursor:"pointer"}}>
      {L.map((l,i)=><span key={i} style={{color:l.c}}>{l.ch}</span>)}
    </span>
  );
}

// ════ APP ════════════════════════════════════════════════════════
export default function App(){
  const [dark,setDark]=useState(false);
  const [query,setQuery]=useState("");
  const [searched,setSearched]=useState(false);
  const [data,setData]=useState(null);
  const [dataB,setDataB]=useState(null);
  const [loading,setLoading]=useState(false);
  const [loadingB,setLoadingB]=useState(false);
  const [newQ,setNewQ]=useState("");
  const [loadDuration,setLoadDuration]=useState(4);
  const inputRef=useRef();

  const P=dark?DARK:LIGHT;
  const MC=dark?MC_D:MC_L;
  const dotColor=dark?"#1E1B4B":"#D8D0F0";

  const doSearch=async(q)=>{
    if(!q.trim()) return;
    const dur=3+Math.random();
    setLoadDuration(dur);
    setQuery(q);setLoading(true);setSearched(true);
    setData(null);setDataB(null);setNewQ("");
    const [res]=await Promise.all([
      callAPI("/predict",{text:q}).catch(()=>null),
      new Promise(r=>setTimeout(r,dur*1000)),
    ]);
    if(res) setData(mapPredict(q,res));
    setLoading(false);
  };

  const doCompare=async()=>{
    if(!query.trim()) return;
    setLoadingB(true);setDataB(null);
    try{setDataB(mapCompare(query,await callAPI("/compare",{text:query})));}catch(e){console.error(e);}
    setLoadingB(false);
  };

  const reset=()=>{setSearched(false);setData(null);setDataB(null);setQuery("");setNewQ("");};

  return(
    <div style={{minHeight:"100vh",background:P.bg,fontFamily:"'Inter','Segoe UI',sans-serif",color:P.ink,transition:"background .3s,color .3s"}}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@700;800;900&family=Inter:wght@300;400;500;600&display=swap');
        *{box-sizing:border-box;margin:0;padding:0;}
        @keyframes qspin{to{transform:rotate(360deg);}}
        @keyframes pulse{0%,100%{opacity:.35;transform:scale(1);}50%{opacity:1;transform:scale(1.4);}}
        @keyframes blink{0%,100%{opacity:.4;}50%{opacity:.08;}}
        @keyframes pingAnim{0%{transform:scale(1);opacity:.6;}100%{transform:scale(2.4);opacity:0;}}
        @keyframes up{from{opacity:0;transform:translateY(13px);}to{opacity:1;transform:translateY(0);}}
        .a1{animation:up .38s .07s ease both;}
        .a2{animation:up .38s .14s ease both;}
        .a3{animation:up .38s .21s ease both;}
        .pill{display:flex;align-items:center;gap:10px;border:1.5px solid ${P.border};border-radius:999px;padding:13px 22px;background:${P.cardBg};box-shadow:0 2px 12px rgba(108,63,197,.07);transition:all .2s;width:100%;}
        .pill:focus-within{border-color:${P.primary}66;box-shadow:0 4px 22px rgba(108,63,197,.13);}
        .pill-sm{padding:8px 17px;}
        .si{flex:1;border:none;outline:none;font-size:16px;color:${P.ink};background:transparent;font-family:'Inter',sans-serif;}
        .si::placeholder{color:${P.subtle};}
        .pbtn{display:inline-flex;align-items:center;gap:8px;background:${P.primary};color:#fff;border:none;border-radius:999px;padding:12px 28px;font-size:14px;font-weight:600;cursor:pointer;font-family:'Inter',sans-serif;box-shadow:0 3px 14px rgba(108,63,197,.3);transition:all .15s;white-space:nowrap;}
        .pbtn:hover{filter:brightness(1.1);box-shadow:0 5px 22px rgba(108,63,197,.42);transform:translateY(-1px);}
        .pbtn:active{transform:scale(.97);}
        .sbtn{display:inline-flex;align-items:center;gap:7px;padding:8px 16px;border-radius:999px;border:1.5px solid ${P.border};background:${P.cardBg};color:${P.ink};font-size:12px;font-weight:500;cursor:pointer;font-family:'Inter',sans-serif;transition:all .15s;}
        .sbtn:hover{border-color:${P.primary}55;background:${P.purpleTint};}
        .sbtn:disabled{opacity:.45;cursor:not-allowed;pointer-events:none;}
        .nav{position:fixed;top:0;left:0;right:0;z-index:300;background:${P.navBg};backdrop-filter:blur(20px);border-bottom:1px solid ${P.border};display:flex;align-items:center;gap:8px;padding:8px 20px;box-shadow:0 1px 10px rgba(108,63,197,.07);}
        .card{background:${P.cardBg};border:1px solid ${P.cardBorder};border-radius:16px;padding:20px 22px;transition:box-shadow .2s,border-color .2s;}
        .card:hover{box-shadow:0 6px 24px rgba(108,63,197,.09);border-color:${P.primary}33;}
        .lbl{font-size:10.5px;font-weight:600;letter-spacing:1.4px;text-transform:uppercase;color:${P.subtle};margin-bottom:12px;font-family:'Inter',sans-serif;}
        .bdg{display:inline-flex;align-items:center;padding:3px 10px;border-radius:999px;font-size:11px;font-weight:600;}
        .ins{display:flex;align-items:flex-start;gap:12px;padding:12px 14px;border-radius:10px;border:1px solid ${P.border};margin-bottom:7px;transition:all .15s;cursor:default;}
        .ins:hover{background:${P.bgSoft};border-color:${P.primary}33;transform:translateX(2px);}
        .g2{display:grid;grid-template-columns:1fr 1fr;gap:14px;}
        .g3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;}
        .g4{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;}
        @media(max-width:860px){.g4{grid-template-columns:1fr 1fr;}}
        @media(max-width:580px){.g2,.g3,.g4{grid-template-columns:1fr;}}
        .dvd{height:1px;background:${P.border};margin:14px 0;}
        ::-webkit-scrollbar{width:5px;}
        ::-webkit-scrollbar-thumb{background:${P.border};border-radius:3px;}
      `}</style>

      {/* ═══ HOME ═══ */}
      {!searched&&(
        <div style={{
          display:"flex",flexDirection:"column",alignItems:"center",justifyContent:"center",
          minHeight:"100vh",padding:24,position:"relative",overflow:"hidden",
          backgroundColor: dark ? "#07050F" : "#FFFFFF",
          backgroundImage: dark
            ? `radial-gradient(${dotColor} 1px, transparent 1px)`
            : `radial-gradient(${dotColor} 1px, transparent 1px), linear-gradient(135deg,#F5F0FF 0%,#EBF9F4 50%,#FFF8ED 100%)`,
          backgroundSize: dark ? "26px 26px" : "26px 26px, 100% 100%",
        }}>
          <div style={{position:"fixed",top:-120,right:-120,width:420,height:420,borderRadius:"50%",background:`radial-gradient(circle,${P.purpleTint} 0%,transparent 70%)`,pointerEvents:"none",zIndex:0}}/>
          <div style={{position:"fixed",bottom:-100,left:-100,width:360,height:360,borderRadius:"50%",background:`radial-gradient(circle,${P.mintTint} 0%,transparent 70%)`,pointerEvents:"none",zIndex:0}}/>

          <div style={{position:"fixed",top:14,right:20,display:"flex",gap:8,alignItems:"center",zIndex:10}}>
            <ServerStatus P={P}/>
            <button className="sbtn" onClick={()=>window.open(`${BASE_URL}/docs`,"_blank")}>
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/><polyline points="14,2 14,8 20,8" stroke="currentColor" strokeWidth="2"/><line x1="16" y1="13" x2="8" y2="13" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/></svg>
              API Docs
            </button>
            <button className="sbtn" onClick={()=>setDark(d=>!d)} style={{padding:"7px 13px"}}>
              {dark?"☀️":"🌙"}
            </button>
            <LiveClock P={P}/>
          </div>

          <div style={{position:"relative",zIndex:1,display:"flex",flexDirection:"column",alignItems:"center",width:"100%",maxWidth:680}}>
            {/* Logo with dark prop */}
            <Logo size={64} dark={dark}/>
            <p style={{marginTop:6,marginBottom:6,fontSize:13,color:P.subtle,letterSpacing:".5px",fontFamily:"'Inter',sans-serif"}}>Sentiment &amp; Mood Intelligence Platform</p>
            <p style={{marginBottom:36,fontSize:11,color:P.subtle,letterSpacing:"1.5px",textTransform:"uppercase",fontFamily:"'Inter',sans-serif"}}>Zomato Dataset · RoBERTa · BERT · ALBERT</p>

            <div style={{width:"100%"}}>
              <div className="pill" style={{padding:"16px 26px"}}>
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none"><circle cx="11" cy="11" r="7.5" stroke={P.subtle} strokeWidth="2"/><path d="M16.5 16.5L21 21" stroke={P.subtle} strokeWidth="2" strokeLinecap="round"/></svg>
                <input ref={inputRef} className="si" style={{fontSize:18,color:P.ink}}
                  placeholder="Search any topic, brand or review…"
                  value={query} onChange={e=>setQuery(e.target.value)}
                  onKeyDown={e=>e.key==="Enter"&&doSearch(query)} autoFocus/>
                {query&&<svg onClick={()=>setQuery("")} width="18" height="18" viewBox="0 0 24 24" fill="none" style={{cursor:"pointer",flexShrink:0}}><circle cx="12" cy="12" r="10" fill={P.subtle}/><path d="M8 8l8 8M16 8l-8 8" stroke="#fff" strokeWidth="2" strokeLinecap="round"/></svg>}
              </div>
            </div>

            <button className="pbtn" style={{marginTop:22}} onClick={()=>doSearch(query)}>
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none"><circle cx="11" cy="11" r="7" stroke="rgba(255,255,255,.9)" strokeWidth="2.2"/><path d="M16.5 16.5L21 21" stroke="rgba(255,255,255,.9)" strokeWidth="2.2" strokeLinecap="round"/></svg>
              Analyse Sentiment
            </button>

            <div style={{marginTop:24,display:"flex",gap:8,flexWrap:"wrap",justifyContent:"center"}}>
              {[`"Amazing food and service!"`,`"Worst experience ever"`,`"Decent but overpriced"`].map((ex,i)=>(
                <button key={i} onClick={()=>doSearch(ex.replace(/"/g,""))} style={{fontSize:11,padding:"5px 14px",borderRadius:999,background:P.cardBg,border:`1.5px solid ${P.border}`,color:P.muted,cursor:"pointer",fontFamily:"'Inter',sans-serif",transition:"all .15s"}}>
                  {ex}
                </button>
              ))}
            </div>

            <div style={{marginTop:48,display:"flex",alignItems:"center",gap:10}}>
              <div style={{width:2,height:20,background:`linear-gradient(${P.primary},${P.mint})`,borderRadius:2}}/>
              <p style={{fontSize:11,color:P.subtle,letterSpacing:"1.5px",textTransform:"uppercase",fontFamily:"'Inter',sans-serif"}}>MoodLens AI · v1.0</p>
              <div style={{width:2,height:20,background:`linear-gradient(${P.mint},${P.primary})`,borderRadius:2}}/>
            </div>
          </div>
        </div>
      )}

      {/* ═══ NAVBAR ═══ */}
      {searched&&(
        <nav className="nav">
          <div onClick={reset} style={{flexShrink:0,cursor:"pointer"}}><Logo size={24} dark={dark}/></div>
          <div style={{flex:1,maxWidth:460}}>
            <div className="pill pill-sm">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none"><circle cx="11" cy="11" r="7.5" stroke={P.subtle} strokeWidth="2"/><path d="M16.5 16.5L21 21" stroke={P.subtle} strokeWidth="2" strokeLinecap="round"/></svg>
              <input className="si" style={{fontSize:13,color:P.ink}} placeholder="Search again…"
                value={newQ} onChange={e=>setNewQ(e.target.value)}
                onKeyDown={e=>e.key==="Enter"&&doSearch(newQ||query)}/>
              {newQ&&<svg onClick={()=>setNewQ("")} width="14" height="14" viewBox="0 0 24 24" fill="none" style={{cursor:"pointer",flexShrink:0}}><circle cx="12" cy="12" r="10" fill={P.subtle}/><path d="M8 8l8 8M16 8l-8 8" stroke="#fff" strokeWidth="2" strokeLinecap="round"/></svg>}
            </div>
          </div>
          <button className="pbtn" style={{padding:"8px 18px",fontSize:12}} onClick={()=>doSearch(newQ||query)}>Search</button>
          <button className="sbtn" onClick={doCompare} disabled={loadingB}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none"><path d="M8 7l-4 5 4 5M16 7l4 5-4 5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/></svg>
            {loadingB?"Running…":"Compare"}
          </button>
          <button className="sbtn" onClick={()=>window.open(`${BASE_URL}/docs`,"_blank")}>
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z" stroke="currentColor" strokeWidth="2"/><polyline points="14,2 14,8 20,8" stroke="currentColor" strokeWidth="2"/></svg>
            API
          </button>
          <button className="sbtn" onClick={()=>exportPDF(data,dataB,query)}>
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none"><path d="M12 3v13M7 11l5 5 5-5" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round"/><path d="M3 20h18" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/></svg>
            PDF
          </button>
          <ServerStatus P={P}/>
          <button onClick={()=>setDark(d=>!d)} style={{background:"none",border:"none",cursor:"pointer",fontSize:16,flexShrink:0,padding:"4px"}}>{dark?"☀️":"🌙"}</button>
          <LiveClock P={P}/>
        </nav>
      )}

      {/* ═══ DASHBOARD ═══ */}
      {searched&&(
        <div style={{maxWidth:1080,margin:"0 auto",padding:"80px 24px 56px"}}>
          {loading&&<QueueLoader query={query} P={P} TOTAL={loadDuration}/>}
          {!loading&&data&&(<>
            <PredictionHero sentiment={data.sentiment} score={data.score} topic={data.topic} P={P}/>
            <div className="g2 a2" style={{marginBottom:14}}>
              <div className="card">
                <p className="lbl">Confidence Breakdown — RoBERTa</p>
                {data.breakdown.map((c,i)=>(
                  <Bar key={i} label={c.label} value={c.value} color={c.color} delay={i*110} P={P}/>
                ))}
                {dataB&&(<>
                  <div className="dvd"/>
                  <p className="lbl" style={{color:P.mint}}>All Models — Confidence</p>
                  {dataB.models.map((m,i)=>(
                    <Bar key={m.model} label={m.model.toUpperCase()} value={Math.round(m.confidence*100)} color={MC[i]} delay={i*130} note={m.prediction} P={P}/>
                  ))}
                </>)}
              </div>
              <div className="card">
                <p className="lbl">Smart Insights</p>
                {data.insights.map((ins,i)=>(
                  <div key={i} className="ins">
                    <span style={{fontSize:18,flexShrink:0,marginTop:1}}>{ins.icon}</span>
                    <span style={{fontSize:13,color:P.muted,flex:1,lineHeight:1.6}}>{ins.text}</span>
                    <span className="bdg" style={{background:dark?`${ins.tc}22`:ins.tb,color:ins.tc,border:`1px solid ${ins.tc}33`,flexShrink:0}}>{ins.tag}</span>
                  </div>
                ))}
                {loadingB&&(
                  <div style={{display:"flex",alignItems:"center",gap:10,marginTop:10,padding:12,background:P.bgSoft,borderRadius:10,border:`1px solid ${P.border}`}}>
                    <div style={{display:"flex",gap:4}}>{MC.map((c,i)=><div key={i} style={{width:8,height:8,borderRadius:"50%",background:c,animation:`pingAnim 1.2s ${i*.18}s ease-out infinite`}}/>)}</div>
                    <span style={{fontSize:12,color:P.subtle}}>Running all 4 models in parallel…</span>
                  </div>
                )}
                {dataB&&(<>
                  <div className="dvd"/>
                  <p className="lbl" style={{color:P.mint,marginBottom:8}}>Multi-Model Insights</p>
                  {dataB.insights.map((ins,i)=>(
                    <div key={i} className="ins">
                      <span style={{fontSize:18,flexShrink:0,marginTop:1}}>{ins.icon}</span>
                      <span style={{fontSize:13,color:P.muted,flex:1,lineHeight:1.6}}>{ins.text}</span>
                      <span className="bdg" style={{background:dark?`${ins.tc}22`:ins.tb,color:ins.tc,border:`1px solid ${ins.tc}33`,flexShrink:0}}>{ins.tag}</span>
                    </div>
                  ))}
                </>)}
              </div>
            </div>
            {dataB&&(
              <div className="a3" style={{marginBottom:14}}>
                <div className="card">
                  <div style={{display:"flex",alignItems:"center",justifyContent:"space-between",marginBottom:16}}>
                    <p className="lbl" style={{margin:0}}>All Models — Confidence Rings</p>
                    <span style={{fontSize:11,color:P.mint,fontWeight:600,padding:"3px 11px",background:P.mintTint,borderRadius:999,border:`1px solid ${P.mint}33`}}>⭐ Best model in green</span>
                  </div>
                  <div className="g4">
                    {dataB.models.map((m,i)=>(
                      <ModelCard key={m.model} model={m.model} confidence={m.confidence} prediction={m.prediction} color={MC[i]} isBest={i===0} delay={i*130+80} P={P}/>
                    ))}
                  </div>
                </div>
              </div>
            )}
            <p style={{textAlign:"center",fontSize:11,color:P.subtle,letterSpacing:".8px",textTransform:"uppercase"}}>
              MoodLens · Zomato Sentiment Intelligence · For informational use only
            </p>
          </>)}
        </div>
      )}
    </div>
  );
}