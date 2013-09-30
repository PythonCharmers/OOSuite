classdef Wormhole
    properties
        conn
    end
    
    methods
        function self = Wormhole
            self.conn = tcpip('localhost',5001,'OutputBufferSize',1024,'InputBufferSize',1024);
            fopen(self.conn);
        end
        
        function put(self,name,val)
            fwrite(self.conn,sprintf('%10s','put'))
            fwrite(self.conn,sprintf('%10s',name))
            fwrite(self.conn,sprintf('%30s',num2str(size(val))))
            n_floats = numel(val);
            n_sent = 0;
            while n_sent < n_floats
                n_tosend = min(128,n_floats-n_sent);
                fwrite(self.conn,val((n_sent+1):n_sent+n_tosend),'double')
                n_sent = n_sent + n_tosend;
%~                fprintf('%i/%i floats sent\n',n_sent,n_floats)
                
            end
        end
        
        function execute(self,stmt)
            fwrite(self.conn,sprintf('%10s','exec'))
            fwrite(self.conn,sprintf('%100s',stmt))
        end
        
        function val = get(self,name)
            fwrite(self.conn,sprintf('%10s','get'))
            fwrite(self.conn,sprintf('%10s',name))
            shape_str = char(fread(self.conn, 30,'char'))';
            shape = eval(shape_str);
            val = zeros(shape);
            n_floats = numel(val);
            n_read = 0;
            while n_read < n_floats
                n_toread = min(128,n_floats-n_read);
                val(n_read+1:n_read+n_toread) = fread(self.conn,n_toread,'double');
                n_read = n_read + n_toread;
%~                fprintf('%i/%i floats read\n',n_read,n_floats)
            end
            
        end
        
    end
end
