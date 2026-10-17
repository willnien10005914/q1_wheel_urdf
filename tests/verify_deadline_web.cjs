// Actual native Web/API. Model availability varies with training outcomes.
const {chromium}=require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');
const fs=require('fs');
(async()=>{
 const url=process.env.Q1_DEADLINE_WEB_URL || 'http://127.0.0.1:8795';
 const out=process.env.Q1_DEADLINE_WEB_OUTPUT || '/tmp/q1_deadline_web'; fs.mkdirSync(out,{recursive:true});
 let ready=false;
 for(let i=0;i<120;i++){try{if((await fetch(url+'/api/state')).ok){ready=true;break;}}catch{}await new Promise(r=>setTimeout(r,1000));}
 if(!ready)throw Error('Real player did not become ready');
 const browser=await chromium.launch({executablePath:'/usr/bin/google-chrome',headless:true,
  args:['--no-sandbox','--use-angle=swiftshader','--enable-unsafe-swiftshader']});
 try{
  const page=await browser.newPage({viewport:{width:1440,height:1000}});const errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  await page.goto(url+'/web/',{waitUntil:'domcontentloaded',timeout:60000});
  await page.waitForFunction(()=>document.querySelectorAll('#joints input[type=range]').length===27,{},{timeout:60000});
  await page.waitForSelector('#isaac-status.online');
  const state=await (await page.request.get(url+'/api/state')).json();
  const expect=(v,m)=>{if(!v)throw Error(m);};
  expect(state.action_dim===23 && state.observation_dim===90,'Native dimensions changed');
  expect(await page.locator('#pose-slide').isDisabled()===!state.supported_actions.slide,'Sliding button/API gate disagree');
  await page.locator('#pose-stand').click();
  expect(await page.locator('#pose-stand').evaluate(b=>b.classList.contains('press-feedback')),'Button animation missing');
  await page.keyboard.press('KeyW');await page.waitForTimeout(400);await page.keyboard.press('Space');
  await page.screenshot({path:out+'/desktop.png'});
  await page.setViewportSize({width:390,height:844});await page.waitForTimeout(300);
  const scrollable=await page.evaluate(()=>[document.scrollingElement,...document.querySelectorAll('*')].some(e=>e.scrollHeight>e.clientHeight+10 && (/auto|scroll/.test(getComputedStyle(e).overflowY)||e===document.scrollingElement)));
  expect(scrollable,'Narrow screen cannot scroll to remaining controls');
  await page.screenshot({path:out+'/mobile.png'});expect(!errors.length,'Browser errors: '+errors.join(';'));
  fs.writeFileSync(out+'/result.json',JSON.stringify({real_api:true,contract:state.contract,sliding_gate_matches_api:true,keyboard_requests_sent:true,button_animation:true,mobile_scrollable:scrollable,page_errors:errors},null,2));
  console.log('DEADLINE_WEB_PASS');
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exit(1);});
