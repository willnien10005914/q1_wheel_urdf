// Real native player/API; no fixture models or fabricated robot success.
const {chromium}=require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');
const fs=require('fs');
(async()=>{
 const url=process.env.Q1_SLIDE_WEB_URL || 'http://127.0.0.1:8793';
 const out=process.env.Q1_SLIDE_WEB_OUTPUT || '/tmp/q1_sliding_web';fs.mkdirSync(out,{recursive:true});
 for(let i=0;i<60;i++) {try {if((await fetch(url+'/api/state')).ok)break;}catch{}await new Promise(r=>setTimeout(r,1000));}
 const browser=await chromium.launch({executablePath:process.env.CHROME_BIN || '/usr/bin/google-chrome',headless:true,
   args:['--no-sandbox','--use-angle=swiftshader','--enable-unsafe-swiftshader']});
 const page=await browser.newPage({viewport:{width:1440,height:1000}});const errors=[];
 page.on('pageerror',e=>errors.push(e.message));
 await page.goto(url+'/web/',{waitUntil:'domcontentloaded',timeout:60000});
 await page.waitForFunction(()=>document.querySelectorAll('#joints input[type=range]').length===27,{},{timeout:60000});
 await page.waitForSelector('#isaac-status.online');
 const expect=(v,m)=>{if(!v)throw Error(m);};
 const button=page.locator('#pose-slide');expect(await button.count()===1,'Missing distinct sliding button');
 expect(await button.isDisabled(),'Untrained sliding must be disabled');
 expect(await button.getAttribute('data-action-state')==='等待模型','Missing waiting-model explanation');
 await page.locator('#pose-stand').click();
 expect(await page.locator('#pose-stand').evaluate(b=>b.classList.contains('press-feedback')),'Press animation regressed');
 await page.keyboard.press('KeyP');await page.waitForTimeout(400);
 const s=await (await page.request.get(url+'/api/state')).json();
 expect(s.environment==='ManagerBasedRLEnv' && s.action_dim===23 && s.observation_dim===90,'Native contract changed');
 expect(s.pose==='stand' && !s.supported_actions.slide,'Unvalidated sliding bypassed server gate');
 expect(s.base.z>.7,'Stand was broken by the sliding environment extension');
 expect(errors.length===0,'Page errors: '+errors.join(';'));
 await page.screenshot({path:out+'/desktop.png'});
 await page.setViewportSize({width:390,height:844});await page.screenshot({path:out+'/mobile.png'});
 const result={real_api:true,new_urdf:true,distinct_sliding_button:true,unvalidated_disabled:true,
   keyboard_cannot_bypass_validation:true,stand_still_upright:true,root_height:s.base.z,page_errors:errors};
 fs.writeFileSync(out+'/result.json',JSON.stringify(result,null,2));console.log('SLIDING_WEB_PASS',JSON.stringify(result));
 await browser.close();
})().catch(e=>{console.error(e);process.exit(1);});
