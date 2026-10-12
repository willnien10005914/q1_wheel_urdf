// UI-only fixtures: intercept API calls; no robot physics is fabricated or changed.
const {chromium}=require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');
const fs=require('fs');
const out=process.env.Q1_UI_TEST_OUTPUT || '/tmp/q1_button_feedback';fs.mkdirSync(out,{recursive:true});
(async()=>{
 const browser=await chromium.launch({executablePath:process.env.CHROME_BIN || '/usr/bin/google-chrome',headless:true,args:['--no-sandbox','--use-angle=swiftshader','--enable-unsafe-swiftshader']});
 const page=await browser.newPage({viewport:{width:1440,height:1000}});const errors=[],commands=[];
 let state={ok:true,pose:'stand',message:'UI fixture: standing',cmd:{vx:0,yaw:0},joints:{},manual_control_available:true,policies:{backflip:true,unbox:true,supine_elbow:true,prone_elbow:true,supine_gripper:true,prone_gripper:true},backflip:{success:false}};
 page.on('pageerror',e=>errors.push(e.message));
 await page.route('**/api/**',async route=>{
  const r=route.request();if(r.url().endsWith('/api/state'))return route.fulfill({json:state});
  if(r.url().endsWith('/api/pose')){const name=r.postDataJSON().name;commands.push(name);await new Promise(r=>setTimeout(r,160));state={...state,pose:name==='kneel'?'kneeling':name};}
  return route.fulfill({json:{ok:true}});
 });
 await page.goto((process.env.Q1_UI_TEST_URL || 'http://127.0.0.1:8789')+'/web/');
 await page.waitForFunction(()=>document.querySelectorAll('#joints input[type=range]').length===26,{},{timeout:60000});
 await page.waitForSelector('#isaac-status.online');
 const expect=(v,m)=>{if(!v)throw Error(m);};
 expect(await page.locator('[data-pose=unbox]').count()===1,'Unbox must have one button');
 expect(await page.locator('[data-pose=stand],[data-pose=recovery]').count()===0,'Duplicate action buttons remain');
 const unbox=page.locator('#pose-unbox');const baseline=await unbox.evaluate(b=>({border:getComputedStyle(b).borderColor,outline:getComputedStyle(b).outlineStyle}));
 await page.locator('#pose-kneel').click();expect(await page.locator('#pose-kneel').evaluate(b=>b.classList.contains('press-feedback')),'Kneel has no press feedback');
 await page.waitForFunction(()=>document.querySelector('#pose-kneel').dataset.actionState==='執行中');
 state={...state,pose:'kneel'};await page.waitForFunction(()=>document.querySelector('#pose-kneel').dataset.actionState==='已跪穩');
 expect(await page.locator('#pose-kneel').getAttribute('aria-busy')==='false','Completed kneel still busy');
 await unbox.click();expect(await unbox.evaluate(b=>b.classList.contains('press-feedback')),'Unbox has no press feedback');
 await page.waitForFunction(()=>document.querySelector('#pose-unbox').dataset.actionState==='執行中');
 expect(await unbox.getAttribute('aria-busy')==='true','Unbox executing state missing');
 await page.screenshot({path:out+'/executing.png'});
 state={...state,pose:'unbox_hold'};await page.waitForFunction(()=>document.querySelector('#pose-unbox').dataset.actionState==='已站穩');
 await page.mouse.move(20,200);await page.waitForTimeout(400);
 expect(await unbox.evaluate(b=>!b.classList.contains('action-running')&&!b.classList.contains('active-pose')),'Completed Unbox still looks selected');
 expect(await unbox.evaluate(b=>getComputedStyle(b).borderColor)===baseline.border,'Unbox has persistent colored border');
 expect(commands.includes('unbox')&&!commands.includes('stand'),'Wrong rise command sent');
 const view=page.locator('[data-view=front]');await view.click();expect(await view.evaluate(b=>b.classList.contains('press-feedback')),'View button uses different click feedback');
 state={...state,pose:'stand',manual_control_available:true};await page.waitForTimeout(100);
 const preset=page.locator('[data-preset=tpose]');await preset.click();expect(await preset.evaluate(b=>b.classList.contains('press-feedback')),'Preset button uses different click feedback');
 const release=page.locator('#follow-all');await release.click();expect(await release.evaluate(b=>b.classList.contains('press-feedback')),'Toolbar button uses different click feedback');
 state={...state,pose:'prepare_backflip'};await page.waitForFunction(()=>document.querySelector('#pose-backflip').dataset.actionState==='準備中');
 state={...state,pose:'backflip',backflip:{success:true}};await page.waitForFunction(()=>document.querySelector('#pose-backflip').dataset.actionState==='已完成');
 expect(await page.locator('#pose-backflip').getAttribute('aria-busy')==='false','Completed backflip still busy');
 await page.setViewportSize({width:390,height:844});await page.evaluate(()=>window.scrollTo(0,350));await page.screenshot({path:out+'/mobile.png'});
 expect(await page.evaluate(()=>window.scrollY)>0,'Mobile scroll regressed');
 expect(errors.length===0,'Page errors: '+errors.join(';'));
 const result={ui_fixtures:true,single_rise_button:true,explicit_recovery_variants:true,unified_press_feedback:true,pending_execution_completion_states:true,no_persistent_green_selection:true,mobile_scroll:true,commands,page_errors:errors};
 fs.writeFileSync(out+'/browser_result.json',JSON.stringify(result,null,2));console.log('BUTTON_FEEDBACK_PASS',JSON.stringify(result));await browser.close();
})().catch(e=>{console.error(e);process.exit(1);});
