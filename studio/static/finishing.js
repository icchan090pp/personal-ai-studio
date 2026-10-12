let finishProject=null, finishAttachment=null, finishTimer=null, finishJob=null;
function finishingComparison(job){const box=node('div',null,'growth-comparison');for(const [title,endpoint] of [['元の画像','before'],['仕上げ後','result']]){const figure=node('figure');const img=node('img');img.src=`/api/jobs/${job.id}/${endpoint}`;img.alt=title;figure.append(node('figcaption',title),img);box.append(figure);}box.append(node('p','Real-ESRGANで画像を高精細化。元の育成モデルの重みは変えていません。','hint comparison-context'));return box;}
function showFinishSource(){const value=$('finish-source').value;const box=$('finish-source-preview');box.replaceChildren();if(!value)return;const [kind,id]=value.split(':');const video=kind==='job'&&jobRows.find(j=>j.id===id)?.kind==='video';const preview=node(video?'video':'img');preview.src=kind==='attachment'?`/api/projects/${finishProject}/finish/attachments/${id}`:`/api/jobs/${id}/result`;if(video){preview.controls=true;preview.preload='metadata';}else preview.alt='仕上げる画像';box.append(preview);}
function openFinishing(jobId=null){if(!current){notice('先にプロジェクトを作成してください。',true);return false;}finishProject=current;finishAttachment=null;finishJob=null;$('finish-source').replaceChildren(new Option('作品を選ぶ / ＋から画像を追加',''));for(const job of jobRows.filter(j=>j.status==='completed'&&['image','video','finish'].includes(j.kind))){$('finish-source').append(new Option(`${job.kind==='video'?'動画の先頭フレーム':'画像'} / ${job.model_name||''} / ${job.created_at}`,`job:${job.id}`));}if(jobId)$('finish-source').value='job:'+jobId;$('finish-status').textContent='元の作品を選ぶか、＋から画像を追加してください。';showFinishSource();$('finish-dialog').showModal();pollFinishing();return true;}
$('finish-button').onclick=()=>openFinishing();
$('close-finish').onclick=()=>{$('finish-dialog').close();clearTimeout(finishTimer);};
$('finish-dialog').addEventListener('close',()=>{clearTimeout(finishTimer);});
$('finish-source').onchange=showFinishSource;
$('finish-add-image').onclick=()=>$('finish-upload').click();
$('add-training-image').onclick=()=>{$($('training-target').value==='image'?'upload':'media-upload').click();};
$('finish-remove-image').onclick=()=>{$('finish-source').value='';$('finish-upload').value='';showFinishSource();};
$('finish-upload').onchange=async()=>{const file=$('finish-upload').files[0];if(!file)return;const pid=finishProject;const form=new FormData();form.append('file',file);try{const added=await api(`/api/projects/${pid}/finish/attachments`,{method:'POST',body:form});if(pid!==finishProject)return;finishAttachment=added.id;$('finish-source').append(new Option(`追加画像 / ${file.name}`,`attachment:${added.id}`));$('finish-source').value='attachment:'+added.id;showFinishSource();$('finish-status').textContent='画像をこのPCに追加しました。仕上げの設定を確認してください。';}catch(e){$('finish-status').textContent=e.message;notice(e.message,true);}finally{$('finish-upload').value='';}};
$('start-finish').onclick=async()=>{const button=$('start-finish');button.disabled=true;finishJob=null;try{if(current!==finishProject)throw new Error('プロジェクトが変わりました。仕上げ画面を開き直してください。');const [kind,id]=$('finish-source').value.split(':');if(!id)throw new Error('仕上げる画像を選択してください。');const created=await api(`/api/projects/${finishProject}/finish`,json({provider:'realesrgan',source_job_id:kind==='job'?id:null,source_attachment_id:kind==='attachment'?id:null,enhancement_model:$('finish-enhancement-model').value,scale:Number($('finish-scale').value),denoise:$('finish-denoise').checked}));finishJob=created.id;$('finish-status').textContent='仕上げを開始しました。完了後はギャラリーで元画像と比較できます。';await refresh();}catch(e){$('finish-status').textContent=e.message;notice(e.message,true);}finally{button.disabled=false;}};
$('stop-finish').onclick=async()=>{try{const reply=await api(`/api/projects/${finishProject}/finish/stop`,json({}));finishJob=null;$('finish-status').textContent=reply.message;await refresh();}catch(e){notice(e.message,true);}};
async function pollFinishing(){clearTimeout(finishTimer);if(!$('finish-dialog').open)return;try{if(current===finishProject){const rows=await api(`/api/projects/${finishProject}/jobs`);if(!finishJob)finishJob=rows.find(j=>j.kind==='finish'&&['queued','running'].includes(j.status))?.id;const latest=rows.find(j=>j.id===finishJob);if(latest){$('finish-status').textContent=latest.status==='failed'?latest.error:latest.status==='completed'?'仕上げ完了。ギャラリーで元画像と比較・保存できます。':`${statusNames[latest.status]} / ${latest.progress}%`;} $('start-finish').disabled=rows.some(j=>['queued','running'].includes(j.status));}}catch(e){$('finish-status').textContent=e.message;}finally{if($('finish-dialog').open)finishTimer=setTimeout(pollFinishing,2500);}}
let generationReference=null, referenceSequence=0;
function updateGenerationReference(){
 const isImage=document.querySelector('[name=kind]:checked')?.value==='image';
 if(generationReference&&generationReference.project!==current){generationReference=null;referenceSequence++;}
 const attached=generationReference?.project===current;
 $('add-generation-image').disabled=!isImage;
 $('generation-reference').hidden=!isImage||!attached;
 $('reference-status').textContent=!isImage?'参考画像は画像生成で使用できます。':attached?'選択中の画像モデルに、この参考画像とプロンプトを反映します。':'';
 if(!attached)$('generation-reference-preview').replaceChildren();
}
function generationReferenceConfig(){
 if(document.querySelector('[name=kind]:checked').value!=='image'||generationReference?.project!==current)return {};
 return {reference_id:generationReference.id,reference_strength:Number($('reference-strength').value)};
}
$('add-generation-image').onclick=()=>{if(!current){notice('先にプロジェクトを作成してください。',true);return;}$('generation-upload').click();};
$('generation-upload').onchange=async()=>{
 const file=$('generation-upload').files[0];if(!file)return;
 const project=current, sequence=++referenceSequence;
 const form=new FormData();form.append('file',file);$('add-generation-image').disabled=true;
 try{const added=await api(`/api/projects/${project}/finish/attachments`,{method:'POST',body:form});if(current!==project||sequence!==referenceSequence)return;
 generationReference={...added,project};const preview=node('img');preview.src=added.url;preview.alt='生成に使う参考画像';$('generation-reference-preview').replaceChildren(preview);notice('参考画像を追加しました。プロンプトと変更の強さを指定して生成できます。');
 }catch(e){notice(e.message,true);}finally{$('generation-upload').value='';updateGenerationReference();}
};
$('remove-generation-image').onclick=()=>{referenceSequence++;generationReference=null;$('generation-upload').value='';updateGenerationReference();};
$('reference-strength').oninput=()=>{$('reference-strength-value').textContent=Number($('reference-strength').value).toFixed(2);};
document.querySelectorAll('[name=kind],#public-model,#audio-model').forEach(el=>el.addEventListener('change',updateGenerationReference));
