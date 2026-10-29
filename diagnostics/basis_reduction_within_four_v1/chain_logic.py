"""Pure ordered-query bookkeeping, shared by scheduling tests and collection."""


def consume_known(group,items,lookup):
    added=[]
    for item in items:
        if group['success']:break
        index=item['index']
        if index<=group['searched']:continue
        assert index==group['searched']+1,('Noncontiguous Oracle search',index,group['searched'])
        result=lookup(item['query_uid'])
        if result is None:break
        query=dict(index=index,query_uid=item['query_uid'],result=result)
        group['queries'].append(query);group['searched']=index;added.append(query)
        if result['success']:group.update(success=True,first_success=index,solution=query)
    return added


def walk(items,evaluate,on_result):
    previous=None
    for item in items:
        if previous is not None:assert item['index']==previous+1
        previous=item['index']
        result=evaluate(item)
        if result is None:return False
        on_result(item,result)
        if result['success']:return True
    return True
