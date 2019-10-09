import boto3
client = boto3.client('lex-runtime')


response = client.post_text(
    botName='insideout',
    botAlias='$LATEST',
    userId='string',
    # sessionAttributes={...}|[...]|123|123.4|'string'|True|None,
    # requestAttributes={...}|[...]|123|123.4|'string'|True|None,
    # contentType='inputText',
    # accept='string',
    inputText='i am sad'
)

print(response)