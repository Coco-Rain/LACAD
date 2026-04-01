import cadquery as cq

result = (
cq.Workplane()
.box(5,5,1)
.faces('>Z')
.sketch()
.regularPolygon(2,3,tag='outer')
.regularPolygon(1.5,3,mode='s')
.vertices(tag='outer')
.chamfer(.2)
.finalize()
.extrude(.5)
)
cq.exporters.export(result, 'GT.stl')